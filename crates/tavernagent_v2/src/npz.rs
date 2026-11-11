//! Minimal `.npz` (zip of `.npy`) reading and writing: f32 / i32 / i64 / u8 / bool arrays.
//!
//! Checkpoints stay readable by numpy (`np.load`), which keeps weights inspectable from
//! Python without a Python implementation of the network.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

const ARRAY_WRITE_BYTES: usize = 64 * 1024;

/// Encode bounded chunks before passing them to the ZIP writer. Buffering only the
/// underlying file still makes ZIP/CRC process each scalar separately. Keep little-
/// endian encoding explicit on every platform and avoid a second array-sized buffer.
fn write_array_chunks<T, const N: usize>(
    w: &mut impl Write,
    values: &[T],
    encode: impl Fn(&T) -> [u8; N],
) -> std::io::Result<()> {
    let mut bytes = [0u8; ARRAY_WRITE_BYTES];
    for chunk in values.chunks(ARRAY_WRITE_BYTES / N) {
        let used = &mut bytes[..chunk.len() * N];
        for (value, dst) in chunk.iter().zip(used.as_chunks_mut::<N>().0) {
            dst.copy_from_slice(&encode(value));
        }
        w.write_all(used)?;
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub enum NpyData {
    F32(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    Bool(Vec<bool>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: NpyData,
}

impl NpyArray {
    pub fn f32(shape: Vec<usize>, data: Vec<f32>) -> NpyArray {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        NpyArray {
            shape,
            data: NpyData::F32(data),
        }
    }
    pub fn i32(shape: Vec<usize>, data: Vec<i32>) -> NpyArray {
        NpyArray {
            shape,
            data: NpyData::I32(data),
        }
    }
    pub fn i64(shape: Vec<usize>, data: Vec<i64>) -> NpyArray {
        NpyArray {
            shape,
            data: NpyData::I64(data),
        }
    }
    pub fn u8(shape: Vec<usize>, data: Vec<u8>) -> NpyArray {
        NpyArray {
            shape,
            data: NpyData::U8(data),
        }
    }
    pub fn bool(shape: Vec<usize>, data: Vec<bool>) -> NpyArray {
        NpyArray {
            shape,
            data: NpyData::Bool(data),
        }
    }
    pub fn scalar_i64(v: i64) -> NpyArray {
        NpyArray::i64(vec![], vec![v])
    }
    pub fn text(s: &str) -> NpyArray {
        NpyArray::u8(vec![s.len()], s.as_bytes().to_vec())
    }

    pub fn as_f32(&self) -> Result<&[f32], String> {
        match &self.data {
            NpyData::F32(v) => Ok(v),
            _ => Err("expected a float32 array".into()),
        }
    }
    pub fn as_i32(&self) -> Result<&[i32], String> {
        match &self.data {
            NpyData::I32(v) => Ok(v),
            _ => Err("expected an int32 array".into()),
        }
    }
    pub fn as_i64(&self) -> Result<&[i64], String> {
        match &self.data {
            NpyData::I64(v) => Ok(v),
            _ => Err("expected an int64 array".into()),
        }
    }
    pub fn as_u8(&self) -> Result<&[u8], String> {
        match &self.data {
            NpyData::U8(v) => Ok(v),
            _ => Err("expected a uint8 array".into()),
        }
    }
    pub fn as_bool(&self) -> Result<&[bool], String> {
        match &self.data {
            NpyData::Bool(v) => Ok(v),
            _ => Err("expected a bool array".into()),
        }
    }
    pub fn as_text(&self) -> Result<String, String> {
        String::from_utf8(self.as_u8()?.to_vec()).map_err(|e| e.to_string())
    }

    fn descr(&self) -> &'static str {
        match self.data {
            NpyData::F32(_) => "<f4",
            NpyData::I32(_) => "<i4",
            NpyData::I64(_) => "<i8",
            NpyData::U8(_) => "|u1",
            NpyData::Bool(_) => "|b1",
        }
    }

    fn write_npy<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        let shape = match self.shape.len() {
            0 => "()".to_string(),
            1 => format!("({},)", self.shape[0]),
            _ => format!(
                "({})",
                self.shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        let mut header = format!(
            "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
            self.descr(),
            shape
        );
        // Magic (6) + version (2) + header length (2) + header, padded to 64 bytes with '\n'.
        let total = 10 + header.len() + 1;
        let pad = (64 - total % 64) % 64;
        header.extend(std::iter::repeat_n(' ', pad));
        header.push('\n');
        w.write_all(b"\x93NUMPY\x01\x00")?;
        w.write_all(&(header.len() as u16).to_le_bytes())?;
        w.write_all(header.as_bytes())?;
        match &self.data {
            NpyData::F32(v) => write_array_chunks(w, v, |x| x.to_le_bytes())?,
            NpyData::I32(v) => write_array_chunks(w, v, |x| x.to_le_bytes())?,
            NpyData::I64(v) => write_array_chunks(w, v, |x| x.to_le_bytes())?,
            NpyData::U8(v) => w.write_all(v)?,
            NpyData::Bool(v) => write_array_chunks(w, v, |x| [u8::from(*x)])?,
        }
        Ok(())
    }

    fn read_npy(bytes: &[u8]) -> Result<NpyArray, String> {
        if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
            return Err("not an npy array".into());
        }
        let (header_len, start) = match bytes[6] {
            1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
            2 | 3 => (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            ),
            v => return Err(format!("unsupported npy version {v}")),
        };
        let header = std::str::from_utf8(&bytes[start..start + header_len])
            .map_err(|_| "npy header is not utf-8")?;
        let descr = field(header, "'descr'").ok_or("npy header has no descr")?;
        let descr = descr.trim().trim_matches(|c| c == '\'' || c == '"');
        if field(header, "'fortran_order'").is_some_and(|f| f.trim().starts_with("True")) {
            return Err("fortran-order arrays are not supported".into());
        }
        let shape_text = field(header, "'shape'").ok_or("npy header has no shape")?;
        let shape_text = shape_text.trim();
        let inner = shape_text
            .strip_prefix('(')
            .and_then(|s| s.strip_suffix(')'))
            .ok_or("bad npy shape")?;
        let shape: Vec<usize> = inner
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<usize>().map_err(|_| "bad npy shape".to_string()))
            .collect::<Result<_, _>>()?;
        let n: usize = shape.iter().product();
        let body = &bytes[start + header_len..];
        let data = match descr {
            "<f4" => NpyData::F32(chunks(body, n, 4, |b| {
                f32::from_le_bytes(b.try_into().unwrap())
            })?),
            "<i4" => NpyData::I32(chunks(body, n, 4, |b| {
                i32::from_le_bytes(b.try_into().unwrap())
            })?),
            "<i8" => NpyData::I64(chunks(body, n, 8, |b| {
                i64::from_le_bytes(b.try_into().unwrap())
            })?),
            "|u1" => NpyData::U8(chunks(body, n, 1, |b| b[0])?),
            "|b1" => NpyData::Bool(chunks(body, n, 1, |b| b[0] != 0)?),
            other => return Err(format!("unsupported npy dtype {other}")),
        };
        Ok(NpyArray { shape, data })
    }
}

fn field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let at = header.find(key)? + key.len();
    let rest = header[at..].trim_start().strip_prefix(':')?;
    let rest = rest.trim_start();
    // A value ends at the next top-level comma.
    let mut depth = 0;
    for (i, c) in rest.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => return Some(&rest[..i]),
            '}' if depth == 0 => return Some(&rest[..i]),
            _ => {}
        }
    }
    Some(rest)
}

fn chunks<T>(body: &[u8], n: usize, size: usize, f: impl Fn(&[u8]) -> T) -> Result<Vec<T>, String> {
    if body.len() < n * size {
        return Err("npy data is shorter than its shape".into());
    }
    Ok(body[..n * size].chunks_exact(size).map(f).collect())
}

/// Create the parent directory of `path` when it has one.
pub fn ensure_parent(path: &Path) -> Result<(), String> {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))
        }
        _ => Ok(()),
    }
}

/// Named arrays, written in name order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Npz {
    pub arrays: BTreeMap<String, NpyArray>,
}

impl Npz {
    pub fn new() -> Npz {
        Npz::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, array: NpyArray) {
        self.arrays.insert(name.into(), array);
    }

    pub fn get(&self, name: &str) -> Result<&NpyArray, String> {
        self.arrays
            .get(name)
            .ok_or_else(|| format!("checkpoint has no array {name:?}"))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.arrays.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.arrays.keys()
    }

    /// Write atomically: to `<path>.tmp`, then rename over `path`.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let temporary = path.with_file_name(format!(
            "{}.tmp",
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("checkpoint")
        ));
        {
            let file =
                File::create(&temporary).map_err(|e| format!("{}: {e}", temporary.display()))?;
            let mut zip = ZipWriter::new(BufWriter::new(file));
            let options = SimpleFileOptions::default()
                .compression_method(CompressionMethod::Stored)
                .large_file(true);
            for (name, array) in &self.arrays {
                zip.start_file(format!("{name}.npy"), options)
                    .map_err(|e| e.to_string())?;
                array.write_npy(&mut zip).map_err(|e| e.to_string())?;
            }
            zip.finish()
                .map_err(|e| e.to_string())?
                .flush()
                .map_err(|e| e.to_string())?;
        }
        std::fs::rename(&temporary, path).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Npz, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut zip = ZipArchive::new(BufReader::new(file)).map_err(|e| e.to_string())?;
        let mut out = Npz::new();
        for i in 0..zip.len() {
            let mut entry = zip.by_index(i).map_err(|e| e.to_string())?;
            let name = entry.name().to_string();
            let Some(stem) = name.strip_suffix(".npy") else {
                continue;
            };
            let mut bytes = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
            let array = NpyArray::read_npy(&bytes).map_err(|e| format!("{name}: {e}"))?;
            out.arrays.insert(stem.to_string(), array);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_payloads_preserve_scalar_bytes_and_partial_writes() {
        // Include a tail after a 64 KiB boundary for every dtype, plus empty/scalar arrays.
        // Compare against scalar little-endian encoding, including noncanonical NaNs.
        struct ShortWriter(Vec<u8>);
        impl Write for ShortWriter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                let n = bytes.len().min(113);
                self.0.extend_from_slice(&bytes[..n]);
                Ok(n)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let float_bits = [
            0,
            0x8000_0000,
            1,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_1234,
            0x7f80_1234,
        ];
        for n in [0, 1, ARRAY_WRITE_BYTES + 3] {
            let shape = if n == 1 { vec![] } else { vec![n] };
            let arrays = [
                NpyArray::f32(
                    shape.clone(),
                    (0..n)
                        .map(|i| f32::from_bits(float_bits[i % float_bits.len()]))
                        .collect(),
                ),
                NpyArray::i32(
                    shape.clone(),
                    (0..n).map(|i| [i32::MIN, -1, 0, i32::MAX][i % 4]).collect(),
                ),
                NpyArray::i64(
                    shape.clone(),
                    (0..n).map(|i| [i64::MIN, -1, 0, i64::MAX][i % 4]).collect(),
                ),
                NpyArray::u8(shape.clone(), (0..n).map(|i| i as u8).collect()),
                NpyArray::bool(shape, (0..n).map(|i| i % 3 == 0).collect()),
            ];
            for array in arrays {
                let expected: Vec<u8> = match &array.data {
                    NpyData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
                    NpyData::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
                    NpyData::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
                    NpyData::U8(v) => v.clone(),
                    NpyData::Bool(v) => v.iter().map(|x| u8::from(*x)).collect(),
                };
                let mut writer = ShortWriter(Vec::new());
                array.write_npy(&mut writer).unwrap();
                let bytes = writer.0;
                let body = 10 + u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
                assert_eq!(&bytes[body..], &expected, "{} length {n}", array.descr());
                let decoded = NpyArray::read_npy(&bytes).unwrap();
                assert_eq!(decoded.shape, array.shape);
                assert_eq!(decoded.descr(), array.descr());
                if let NpyData::F32(v) = &array.data {
                    assert_eq!(
                        decoded
                            .as_f32()
                            .unwrap()
                            .iter()
                            .map(|x| x.to_bits())
                            .collect::<Vec<_>>(),
                        v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    );
                } else {
                    assert_eq!(decoded, array);
                }
            }
        }
    }

    #[test]
    fn chunked_payload_propagates_a_tail_write_failure() {
        let array = NpyArray::i64(
            vec![ARRAY_WRITE_BYTES / 8 + 1],
            vec![42; ARRAY_WRITE_BYTES / 8 + 1],
        );
        let mut complete = Vec::new();
        array.write_npy(&mut complete).unwrap();
        // Enough for the header and first chunk, but not the final element.
        let mut buffer = vec![0u8; complete.len() - 1];
        let err = array.write_npy(&mut buffer.as_mut_slice()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::WriteZero);
    }

    #[test]
    fn arrays_round_trip_through_a_zip() {
        let dir = std::env::temp_dir().join(format!("tavernagent-npz-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.npz");
        let mut npz = Npz::new();
        npz.insert(
            "f",
            NpyArray::f32(vec![2, 3], (0..6).map(|i| i as f32 * 0.25).collect()),
        );
        npz.insert("i", NpyArray::i32(vec![4], vec![-1, 0, 7, i32::MAX]));
        npz.insert("l", NpyArray::scalar_i64(42));
        npz.insert("b", NpyArray::bool(vec![3], vec![true, false, true]));
        npz.insert("t", NpyArray::text("hello, {world}"));
        npz.save(&path).unwrap();
        let back = Npz::load(&path).unwrap();
        assert_eq!(back, npz);
        assert_eq!(back.get("t").unwrap().as_text().unwrap(), "hello, {world}");
        assert_eq!(back.get("l").unwrap().as_i64().unwrap(), &[42]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
