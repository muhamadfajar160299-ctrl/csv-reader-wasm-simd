//! Fast CSV reader: parses CSV text (UTF-16 code units) exactly like the
//! csv-reader 1.0.0 reference, using WASM SIMD128 to scan for delimiters.
//!
//! The parser works on UTF-16 code units (the same units a JS string is made
//! of), so no re-encoding is ever needed and every code unit survives the
//! round trip, including lone surrogates. Plain (unquoted) fields are
//! returned as spans into the original text; only quoted fields (which need
//! unescaping and CR/LF normalisation) are materialised into an output
//! buffer. All scanning for delimiters is done 8 units at a time with
//! 128-bit SIMD compares.

use core::arch::wasm32::*;
use std::sync::Mutex;

// ---- error codes (mapped to names by solve.js) ----
const E_STRAY_CR: u32 = 1;
const E_QUOTE_JUNK: u32 = 2;
const E_UNTERMINATED_QUOTE: u32 = 3;
const E_SHORT_ROW: u32 = 4;
const E_LONG_ROW: u32 = 5;

// ---- field flags ----
const F_TRANSFORMED: u32 = 1;

const NULL_ROW: u32 = u32::MAX;

const Q: u16 = 0x22; // "
const COMMA: u16 = 0x2C; // ,
const LF: u16 = 0x0A;
const CR: u16 = 0x0D;
const BOM: u16 = 0xFEFF;

#[repr(C)]
#[derive(Clone, Copy)]
struct FieldEntry {
    start: u32,
    len: u32,
    flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RecordEntry {
    field_start: u32,
    field_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ErrorEntry {
    code: u32,
    line: u32,
    row: u32,
    rec_idx: u32,
}

// Reusable scratch buffers, kept alive across calls so JS can write the
// input into linear memory and read the serialised result back out.
static INPUT_BUF: Mutex<Vec<u16>> = Mutex::new(Vec::new());
static OUT_BUF: Mutex<Vec<u8>> = Mutex::new(Vec::new());

/// Ensure the input scratch buffer holds at least `n` UTF-16 units and
/// return its address. JS copies the text bytes there.
#[no_mangle]
pub extern "C" fn input_buf(n: u32) -> *mut u16 {
    let mut b = INPUT_BUF.lock().unwrap();
    b.resize(n as usize, 0);
    b.as_mut_ptr()
}

/// Parse the text already placed at `in_ptr` (`in_len` UTF-16 units).
/// `header` is 1 when the first record is a header row. Returns a pointer
/// to the serialised result blob:
///   u32[0] field_count, u32[1] record_count, u32[2] error_count,
///   u32[3] out_len (UTF-16 units of transformed content),
///   u32[4] fields_off, u32[5] records_off, u32[6] errors_off,
///   u32[7] content_off (all offsets are bytes from the blob start).
/// Then follow: field table (12 bytes/entry: start, len, flags),
/// record table (8 bytes/entry: field_start, field_count),
/// error table (16 bytes/entry: code, line, row, rec_idx),
/// transformed content (out_len UTF-16 units).
#[no_mangle]
pub extern "C" fn parse(in_ptr: *const u16, in_len: u32, header: u32) -> *mut u8 {
    let text = unsafe { core::slice::from_raw_parts(in_ptr, in_len as usize) };
    let mut p = Parser::new(text, header != 0);
    p.run();

    let field_count = p.fields.len() as u32;
    let record_count = p.records.len() as u32;
    let error_count = p.errors.len() as u32;
    let out_len = p.out.len() as u32;
    let fields_off = 32u32;
    let records_off = fields_off + field_count * 12;
    let errors_off = records_off + record_count * 8;
    let content_off = errors_off + error_count * 16;

    let mut blob: Vec<u8> = Vec::with_capacity(
        32 + p.fields.len() * 12 + p.records.len() * 8 + p.errors.len() * 16 + p.out.len() * 2,
    );
    for v in [
        field_count,
        record_count,
        error_count,
        out_len,
        fields_off,
        records_off,
        errors_off,
        content_off,
    ] {
        blob.extend_from_slice(&v.to_le_bytes());
    }
    unsafe {
        let fb = core::slice::from_raw_parts(p.fields.as_ptr() as *const u8, p.fields.len() * 12);
        blob.extend_from_slice(fb);
        let rb = core::slice::from_raw_parts(p.records.as_ptr() as *const u8, p.records.len() * 8);
        blob.extend_from_slice(rb);
        let eb = core::slice::from_raw_parts(p.errors.as_ptr() as *const u8, p.errors.len() * 16);
        blob.extend_from_slice(eb);
        let cb = core::slice::from_raw_parts(p.out.as_ptr() as *const u8, p.out.len() * 2);
        blob.extend_from_slice(cb);
    }
    let mut ob = OUT_BUF.lock().unwrap();
    *ob = blob;
    ob.as_mut_ptr()
}

/// Find the first occurrence of `a`, `b` or `c` in an exactly-8-unit chunk.
/// SIMD: three splat-compares, one bitmask each, OR them, count trailing
/// zeroes to get the first matching lane.
#[target_feature(enable = "simd128")]
unsafe fn find_first_of_3(chunk: &[u16], a: u16, b: u16, c: u16) -> Option<usize> {
    debug_assert_eq!(chunk.len(), 8);
    let v = v128_load(chunk.as_ptr() as *const v128);
    let m = i16x8_bitmask(i16x8_eq(v, i16x8_splat(a as i16)))
        | i16x8_bitmask(i16x8_eq(v, i16x8_splat(b as i16)))
        | i16x8_bitmask(i16x8_eq(v, i16x8_splat(c as i16)));
    if m != 0 {
        Some((m as u32).trailing_zeros() as usize)
    } else {
        None
    }
}

#[derive(Clone, Copy, PartialEq)]
enum FieldEnd {
    Comma,
    Eol,
    Eof,
}

struct Parser<'a> {
    text: &'a [u16],
    pos: usize,
    line: u32,
    out: Vec<u16>,
    fields: Vec<FieldEntry>,
    records: Vec<RecordEntry>,
    errors: Vec<ErrorEntry>,
    header: bool,
    data_rows_done: u32,
    width: u32,
}

impl<'a> Parser<'a> {
    fn new(text: &'a [u16], header: bool) -> Self {
        Parser {
            text,
            pos: 0,
            line: 1,
            out: Vec::new(),
            fields: Vec::new(),
            records: Vec::new(),
            errors: Vec::new(),
            header,
            data_rows_done: 0,
            width: 0,
        }
    }

    /// Row index of the record currently being built (u32::MAX means the
    /// header record, whose errors carry row null).
    fn cur_row(&self) -> u32 {
        if self.header && self.records.is_empty() {
            NULL_ROW
        } else {
            self.data_rows_done
        }
    }

    fn run(&mut self) {
        let n = self.text.len();
        if n > 0 && self.text[0] == BOM {
            self.pos = 1;
        }
        while self.pos < n {
            let c = self.text[self.pos];
            if c == LF {
                self.pos += 1;
                self.line += 1;
                continue;
            }
            if c == CR {
                if self.pos + 1 < n && self.text[self.pos + 1] == LF {
                    self.pos += 2;
                    self.line += 1;
                    continue;
                }
                // Lone CR at the start of a line: dropped, stray_cr, and it
                // belongs to the record that is about to start.
                let row = self.cur_row();
                let rec_idx = self.records.len() as u32;
                self.errors.push(ErrorEntry {
                    code: E_STRAY_CR,
                    line: self.line,
                    row,
                    rec_idx,
                });
                self.pos += 1;
                continue;
            }
            self.parse_record();
        }
    }

    fn parse_record(&mut self) {
        let rec_line = self.line;
        let rec_idx = self.records.len() as u32;
        let row = self.cur_row();
        let field_start = self.fields.len() as u32;
        let mut count: u32 = 0;
        loop {
            let end = self.parse_field(rec_line, row, rec_idx);
            count += 1;
            match end {
                FieldEnd::Comma => continue,
                _ => break,
            }
        }
        self.records.push(RecordEntry {
            field_start,
            field_count: count,
        });
        if self.records.len() == 1 {
            self.width = count;
        } else if count < self.width {
            self.errors.push(ErrorEntry {
                code: E_SHORT_ROW,
                line: rec_line,
                row,
                rec_idx,
            });
        } else if count > self.width {
            self.errors.push(ErrorEntry {
                code: E_LONG_ROW,
                line: rec_line,
                row,
                rec_idx,
            });
        }
        if !(self.header && self.records.len() == 1) {
            self.data_rows_done += 1;
        }
    }

    fn parse_field(&mut self, rec_line: u32, row: u32, rec_idx: u32) -> FieldEnd {
        if self.pos < self.text.len() && self.text[self.pos] == Q {
            return self.parse_quoted_field(rec_line, row, rec_idx);
        }
        self.parse_unquoted_field(rec_line, row, rec_idx)
    }

    fn finish_field(&mut self, start: usize, end: usize, out_start: Option<usize>) {
        match out_start {
            Some(os) => self.fields.push(FieldEntry {
                start: os as u32,
                len: (self.out.len() - os) as u32,
                flags: F_TRANSFORMED,
            }),
            None => self.fields.push(FieldEntry {
                start: start as u32,
                len: (end - start) as u32,
                flags: 0,
            }),
        }
    }

    fn push_transformed(&mut self, out_start: usize) {
        self.fields.push(FieldEntry {
            start: out_start as u32,
            len: (self.out.len() - out_start) as u32,
            flags: F_TRANSFORMED,
        });
    }

    /// Unquoted field: scan for ',' / LF / CR, 8 units at a time with SIMD.
    /// A lone CR is dropped (stray_cr); if any CR was dropped the field is
    /// materialised into the output buffer, otherwise it stays a plain span.
    fn parse_unquoted_field(&mut self, rec_line: u32, row: u32, rec_idx: u32) -> FieldEnd {
        let start = self.pos;
        let n = self.text.len();
        let mut out_start: Option<usize> = None;
        let mut i = self.pos;
        loop {
            if i + 8 <= n {
                let off = unsafe { find_first_of_3(&self.text[i..i + 8], COMMA, LF, CR) };
                match off {
                    Some(o) => {
                        let p = i + o;
                        let c = self.text[p];
                        if c == COMMA {
                            if out_start.is_some() {
                                self.out.extend_from_slice(&self.text[i..p]);
                            }
                            self.finish_field(start, p, out_start);
                            self.pos = p + 1;
                            return FieldEnd::Comma;
                        } else if c == LF {
                            if out_start.is_some() {
                                self.out.extend_from_slice(&self.text[i..p]);
                            }
                            self.finish_field(start, p, out_start);
                            self.pos = p + 1;
                            self.line += 1;
                            return FieldEnd::Eol;
                        } else {
                            // CR
                            if p + 1 < n && self.text[p + 1] == LF {
                                if out_start.is_some() {
                                    self.out.extend_from_slice(&self.text[i..p]);
                                }
                                self.finish_field(start, p, out_start);
                                self.pos = p + 2;
                                self.line += 1;
                                return FieldEnd::Eol;
                            }
                            // Lone CR: dropped, stray_cr.
                            if out_start.is_none() {
                                let os = self.out.len();
                                self.out.extend_from_slice(&self.text[start..p]);
                                out_start = Some(os);
                            } else {
                                self.out.extend_from_slice(&self.text[i..p]);
                            }
                            self.errors.push(ErrorEntry {
                                code: E_STRAY_CR,
                                line: rec_line,
                                row,
                                rec_idx,
                            });
                            i = p + 1;
                            continue;
                        }
                    }
                    None => {
                        if out_start.is_some() {
                            self.out.extend_from_slice(&self.text[i..i + 8]);
                        }
                        i += 8;
                        continue;
                    }
                }
            } else {
                // Scalar tail.
                while i < n {
                    let c = self.text[i];
                    if c == COMMA {
                        self.finish_field(start, i, out_start);
                        self.pos = i + 1;
                        return FieldEnd::Comma;
                    }
                    if c == LF {
                        self.finish_field(start, i, out_start);
                        self.pos = i + 1;
                        self.line += 1;
                        return FieldEnd::Eol;
                    }
                    if c == CR {
                        if i + 1 < n && self.text[i + 1] == LF {
                            self.finish_field(start, i, out_start);
                            self.pos = i + 2;
                            self.line += 1;
                            return FieldEnd::Eol;
                        }
                        if out_start.is_none() {
                            let os = self.out.len();
                            self.out.extend_from_slice(&self.text[start..i]);
                            out_start = Some(os);
                        }
                        self.errors.push(ErrorEntry {
                            code: E_STRAY_CR,
                            line: rec_line,
                            row,
                            rec_idx,
                        });
                        i += 1;
                        continue;
                    }
                    if out_start.is_some() {
                        self.out.push(c);
                    }
                    i += 1;
                }
                self.finish_field(start, n, out_start);
                self.pos = n;
                return FieldEnd::Eof;
            }
        }
    }

    /// Quoted field: find the closing quote (handling "" escapes), copy the
    /// content segments into the output buffer, then consume trailing junk.
    fn parse_quoted_field(&mut self, rec_line: u32, row: u32, rec_idx: u32) -> FieldEnd {
        let n = self.text.len();
        let open_pos = self.pos;
        self.pos += 1; // consume opening quote
        let line_before = self.line;
        let out_start = self.out.len();
        let mut seg_start = self.pos;
        loop {
            let (q, has_cr) = unsafe { self.scan_quoted(seg_start) };
            match q {
                Some(qpos) => {
                    if has_cr {
                        self.copy_cr_normalized(seg_start, qpos);
                    } else {
                        self.out.extend_from_slice(&self.text[seg_start..qpos]);
                    }
                    if qpos + 1 < n && self.text[qpos + 1] == Q {
                        // Escaped quote: one literal quote, keep scanning.
                        self.out.push(Q);
                        self.pos = qpos + 2;
                        seg_start = self.pos;
                        continue;
                    } else {
                        // Closing quote.
                        self.pos = qpos + 1;
                        break;
                    }
                }
                None => {
                    // Unterminated: the opening quote is a literal and the
                    // rest is re-read as an unquoted field. Undo the line
                    // counting the failed scan did; the re-read counts the
                    // breaks again as record terminators.
                    self.out.truncate(out_start);
                    self.line = line_before;
                    self.errors.push(ErrorEntry {
                        code: E_UNTERMINATED_QUOTE,
                        line: rec_line,
                        row,
                        rec_idx,
                    });
                    return self.parse_unterminated(open_pos, rec_line, row, rec_idx);
                }
            }
        }
        self.parse_junk(rec_line, row, rec_idx, out_start)
    }

    /// Scan quoted content from `start` for the next `"`, counting line
    /// breaks (LF, CRLF and lone CR each count once) in the content scanned.
    /// Returns (quote position or None at EOF, whether any CR was seen in
    /// the scanned content).
    #[target_feature(enable = "simd128")]
    unsafe fn scan_quoted(&mut self, start: usize) -> (Option<usize>, bool) {
        let n = self.text.len();
        let mut i = start;
        let mut has_cr = false;
        let vq = i16x8_splat(Q as i16);
        let vn = i16x8_splat(LF as i16);
        let vr = i16x8_splat(CR as i16);
        while i + 8 <= n {
            let v = v128_load(self.text.as_ptr().add(i) as *const v128);
            let qm = i16x8_bitmask(i16x8_eq(v, vq)) as u32;
            let nm = i16x8_bitmask(i16x8_eq(v, vn)) as u32;
            let rm = i16x8_bitmask(i16x8_eq(v, vr)) as u32;
            if qm != 0 {
                let b = qm.trailing_zeros() as usize;
                if b > 0 {
                    let valid = (1u32 << b) - 1;
                    let nm2 = nm & valid;
                    let rm2 = rm & valid;
                    has_cr |= rm2 != 0;
                    let lone = rm2 & !(nm2 >> 1);
                    self.line += nm2.count_ones() + lone.count_ones();
                }
                return (Some(i + b), has_cr);
            }
            // Full chunk, no quote: count its line breaks. A CR in the last
            // lane is only a lone CR if the next unit is not LF.
            has_cr |= rm != 0;
            let mut lone = rm & !(nm >> 1);
            if (rm & 0x80) != 0 && i + 8 < n && self.text[i + 8] == LF {
                lone &= !0x80;
            }
            self.line += nm.count_ones() + lone.count_ones();
            i += 8;
        }
        // Scalar tail.
        while i < n {
            let c = self.text[i];
            if c == Q {
                return (Some(i), has_cr);
            }
            if c == LF {
                self.line += 1;
                i += 1;
                continue;
            }
            if c == CR {
                has_cr = true;
                if i + 1 < n && self.text[i + 1] == LF {
                    self.line += 1;
                    i += 2;
                    continue;
                }
                self.line += 1;
                i += 1;
                continue;
            }
            i += 1;
        }
        (None, has_cr)
    }

    /// Copy text[a..b] into the output buffer, converting CRLF and lone CR
    /// to a single LF. Only used for quoted segments that contain a CR.
    fn copy_cr_normalized(&mut self, mut a: usize, b: usize) {
        while a < b {
            let c = self.text[a];
            if c == CR {
                // A CR at the end of the segment is lone: the next unit is
                // a quote (or EOF), never LF.
                self.out.push(LF);
                if a + 1 < b && self.text[a + 1] == LF {
                    a += 2;
                } else {
                    a += 1;
                }
            } else {
                self.out.push(c);
                a += 1;
            }
        }
    }

    /// After a closing quote: consume whatever follows up to ',' or the end
    /// of the line, appending it as-is (a lone CR is still dropped with a
    /// stray_cr). A quote_junk error is recorded when the field ends, if any
    /// junk was appended.
    fn parse_junk(&mut self, rec_line: u32, row: u32, rec_idx: u32, out_start: usize) -> FieldEnd {
        let n = self.text.len();
        let mut junk_any = false;
        let mut i = self.pos;
        loop {
            if i + 8 <= n {
                let off = unsafe { find_first_of_3(&self.text[i..i + 8], COMMA, LF, CR) };
                match off {
                    Some(o) => {
                        let p = i + o;
                        let c = self.text[p];
                        if p > i {
                            self.out.extend_from_slice(&self.text[i..p]);
                            junk_any = true;
                        }
                        if c == COMMA {
                            if junk_any {
                                self.errors.push(ErrorEntry {
                                    code: E_QUOTE_JUNK,
                                    line: rec_line,
                                    row,
                                    rec_idx,
                                });
                            }
                            self.push_transformed(out_start);
                            self.pos = p + 1;
                            return FieldEnd::Comma;
                        } else if c == LF {
                            if junk_any {
                                self.errors.push(ErrorEntry {
                                    code: E_QUOTE_JUNK,
                                    line: rec_line,
                                    row,
                                    rec_idx,
                                });
                            }
                            self.push_transformed(out_start);
                            self.pos = p + 1;
                            self.line += 1;
                            return FieldEnd::Eol;
                        } else {
                            // CR
                            if p + 1 < n && self.text[p + 1] == LF {
                                if junk_any {
                                    self.errors.push(ErrorEntry {
                                        code: E_QUOTE_JUNK,
                                        line: rec_line,
                                        row,
                                        rec_idx,
                                    });
                                }
                                self.push_transformed(out_start);
                                self.pos = p + 2;
                                self.line += 1;
                                return FieldEnd::Eol;
                            }
                            // Lone CR in junk: dropped, stray_cr; not junk.
                            self.errors.push(ErrorEntry {
                                code: E_STRAY_CR,
                                line: rec_line,
                                row,
                                rec_idx,
                            });
                            i = p + 1;
                            continue;
                        }
                    }
                    None => {
                        self.out.extend_from_slice(&self.text[i..i + 8]);
                        junk_any = true;
                        i += 8;
                        continue;
                    }
                }
            } else {
                while i < n {
                    let c = self.text[i];
                    if c == COMMA {
                        if junk_any {
                            self.errors.push(ErrorEntry {
                                code: E_QUOTE_JUNK,
                                line: rec_line,
                                row,
                                rec_idx,
                            });
                        }
                        self.push_transformed(out_start);
                        self.pos = i + 1;
                        return FieldEnd::Comma;
                    }
                    if c == LF {
                        if junk_any {
                            self.errors.push(ErrorEntry {
                                code: E_QUOTE_JUNK,
                                line: rec_line,
                                row,
                                rec_idx,
                            });
                        }
                        self.push_transformed(out_start);
                        self.pos = i + 1;
                        self.line += 1;
                        return FieldEnd::Eol;
                    }
                    if c == CR {
                        if i + 1 < n && self.text[i + 1] == LF {
                            if junk_any {
                                self.errors.push(ErrorEntry {
                                    code: E_QUOTE_JUNK,
                                    line: rec_line,
                                    row,
                                    rec_idx,
                                });
                            }
                            self.push_transformed(out_start);
                            self.pos = i + 2;
                            self.line += 1;
                            return FieldEnd::Eol;
                        }
                        self.errors.push(ErrorEntry {
                            code: E_STRAY_CR,
                            line: rec_line,
                            row,
                            rec_idx,
                        });
                        i += 1;
                        continue;
                    }
                    self.out.push(c);
                    junk_any = true;
                    i += 1;
                }
                if junk_any {
                    self.errors.push(ErrorEntry {
                        code: E_QUOTE_JUNK,
                        line: rec_line,
                        row,
                        rec_idx,
                    });
                }
                self.push_transformed(out_start);
                self.pos = n;
                return FieldEnd::Eof;
            }
        }
    }

    /// Unterminated quote: the field is '"' plus the rest re-read as an
    /// unquoted field (lone CRs dropped with stray_cr).
    fn parse_unterminated(
        &mut self,
        open_pos: usize,
        rec_line: u32,
        row: u32,
        rec_idx: u32,
    ) -> FieldEnd {
        let out_start = self.out.len();
        self.out.push(Q);
        let n = self.text.len();
        let mut i = open_pos + 1;
        loop {
            if i + 8 <= n {
                let off = unsafe { find_first_of_3(&self.text[i..i + 8], COMMA, LF, CR) };
                match off {
                    Some(o) => {
                        let p = i + o;
                        let c = self.text[p];
                        self.out.extend_from_slice(&self.text[i..p]);
                        if c == COMMA {
                            self.push_transformed(out_start);
                            self.pos = p + 1;
                            return FieldEnd::Comma;
                        } else if c == LF {
                            self.push_transformed(out_start);
                            self.pos = p + 1;
                            self.line += 1;
                            return FieldEnd::Eol;
                        } else {
                            if p + 1 < n && self.text[p + 1] == LF {
                                self.push_transformed(out_start);
                                self.pos = p + 2;
                                self.line += 1;
                                return FieldEnd::Eol;
                            }
                            self.errors.push(ErrorEntry {
                                code: E_STRAY_CR,
                                line: rec_line,
                                row,
                                rec_idx,
                            });
                            i = p + 1;
                            continue;
                        }
                    }
                    None => {
                        self.out.extend_from_slice(&self.text[i..i + 8]);
                        i += 8;
                        continue;
                    }
                }
            } else {
                while i < n {
                    let c = self.text[i];
                    if c == COMMA {
                        self.push_transformed(out_start);
                        self.pos = i + 1;
                        return FieldEnd::Comma;
                    }
                    if c == LF {
                        self.push_transformed(out_start);
                        self.pos = i + 1;
                        self.line += 1;
                        return FieldEnd::Eol;
                    }
                    if c == CR {
                        if i + 1 < n && self.text[i + 1] == LF {
                            self.push_transformed(out_start);
                            self.pos = i + 2;
                            self.line += 1;
                            return FieldEnd::Eol;
                        }
                        self.errors.push(ErrorEntry {
                            code: E_STRAY_CR,
                            line: rec_line,
                            row,
                            rec_idx,
                        });
                        i += 1;
                        continue;
                    }
                    self.out.push(c);
                    i += 1;
                }
                self.push_transformed(out_start);
                self.pos = n;
                return FieldEnd::Eof;
            }
        }
    }
}
