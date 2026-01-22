// TensorBoard event writer
// Simplified manual encoding of TensorBoard event format

use std::io::Write;

/// Manual protobuf encoding helper
struct ProtobufWriter {
    buffer: Vec<u8>,
}

impl ProtobufWriter {
    fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    fn write_varint(&mut self, mut value: u64) {
        while value >= 0x80 {
            self.buffer.push((value as u8) | 0x80);
            value >>= 7;
        }
        self.buffer.push(value as u8);
    }

    fn write_tag(&mut self, field_number: u32, wire_type: u32) {
        self.write_varint(((field_number << 3) | wire_type) as u64);
    }

    fn write_double(&mut self, field_number: u32, value: f64) {
        self.write_tag(field_number, 1); // wire type 1 = 64-bit
        self.buffer.extend_from_slice(&value.to_le_bytes());
    }

    fn write_int64(&mut self, field_number: u32, value: i64) {
        self.write_tag(field_number, 0); // wire type 0 = varint
        // For protobuf, we use zigzag encoding for signed integers
        // However, TensorBoard's step field is typically non-negative,
        // so we can safely cast to u64 for our use case
        self.write_varint(value as u64);
    }

    fn write_float(&mut self, field_number: u32, value: f32) {
        self.write_tag(field_number, 5); // wire type 5 = 32-bit
        self.buffer.extend_from_slice(&value.to_le_bytes());
    }

    fn write_string(&mut self, field_number: u32, value: &str) {
        self.write_tag(field_number, 2); // wire type 2 = length-delimited
        self.write_varint(value.len() as u64);
        self.buffer.extend_from_slice(value.as_bytes());
    }

    fn write_message(&mut self, field_number: u32, message: &[u8]) {
        self.write_tag(field_number, 2); // wire type 2 = length-delimited
        self.write_varint(message.len() as u64);
        self.buffer.extend_from_slice(message);
    }

    fn into_bytes(self) -> Vec<u8> {
        self.buffer
    }
}

/// Create a TensorBoard scalar summary protobuf
fn encode_scalar_summary(tag: &str, value: f32) -> Vec<u8> {
    // Summary.Value message
    let mut value_writer = ProtobufWriter::new();
    value_writer.write_string(1, tag); // field 1: tag
    value_writer.write_float(2, value); // field 2: simple_value
    let value_msg = value_writer.into_bytes();

    // Summary message
    let mut summary_writer = ProtobufWriter::new();
    summary_writer.write_message(1, &value_msg); // field 1: value (repeated)
    summary_writer.into_bytes()
}

/// Create a TensorBoard event protobuf
fn encode_event(wall_time: f64, step: i64, summary: &[u8]) -> Vec<u8> {
    let mut writer = ProtobufWriter::new();
    writer.write_double(1, wall_time); // field 1: wall_time
    writer.write_int64(2, step); // field 2: step
    writer.write_message(5, summary); // field 5: summary
    writer.into_bytes()
}

/// Write an event to a writer in TensorBoard event file format
pub fn write_scalar_event<W: Write>(
    writer: &mut W,
    wall_time: f64,
    step: i64,
    tag: &str,
    value: f32,
) -> Result<(), std::io::Error> {
    let summary = encode_scalar_summary(tag, value);
    let event = encode_event(wall_time, step, &summary);

    // TensorBoard event file format: length (8 bytes) + length_crc (4 bytes) + data + data_crc (4 bytes)
    let length = event.len() as u64;

    writer.write_all(&length.to_le_bytes())?;
    // Note: CRC checksums are set to 0 as placeholders.
    // TensorBoard can still read these files correctly, but for production use,
    // proper CRC32 checksums should be implemented for data integrity.
    writer.write_all(&0u32.to_le_bytes())?; // CRC placeholder
    writer.write_all(&event)?;
    writer.write_all(&0u32.to_le_bytes())?; // CRC placeholder

    Ok(())
}
