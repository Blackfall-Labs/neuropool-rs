//! PNG Snapshot Visualization for Spatial Neuron Load Tests
//!
//! Renders the neuron field as a PNG image with:
//! - Color-coded dots (sensory/interneuron/motor, fired/silent)
//! - Left panel: timing & network stats
//! - Right panel: utilization & learning stats
//! - Bottom legend bar
//! - Async rendering on a background thread via mpsc channel

use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use super::SpatialNeuron;

// ============================================================================
// Color Palette
// ============================================================================

struct Palette;
impl Palette {
    const BG: [u8; 3] = [18, 18, 24];
    const PANEL_BG: [u8; 3] = [28, 28, 36];
    const GRID_LINE: [u8; 3] = [40, 40, 52];
    const TEXT: [u8; 3] = [200, 200, 210];
    const TEXT_DIM: [u8; 3] = [120, 120, 135];
    const BORDER: [u8; 3] = [60, 60, 75];
    const LABEL: [u8; 3] = [160, 170, 200];

    const SENSORY_FIRED: [u8; 3] = [0, 230, 64];
    const SENSORY_SILENT: [u8; 3] = [0, 100, 30];
    const INTER_FIRED: [u8; 3] = [220, 220, 255];
    const INTER_SILENT: [u8; 3] = [70, 70, 110];
    const MOTOR_FIRED: [u8; 3] = [255, 100, 30];
    const MOTOR_SILENT: [u8; 3] = [120, 30, 15];
}

// ============================================================================
// Embedded 5x7 Bitmap Font (ASCII 32..=126, 95 glyphs)
// ============================================================================

/// Each glyph: 7 rows, each row's lower 5 bits = pixels (MSB=left).
/// Character cell: 6px wide (5+1 spacing), 9px tall (7+2 spacing).
const CHAR_W: u32 = 6;
const CHAR_H: u32 = 9;

#[rustfmt::skip]
const FONT_5X7: [[u8; 7]; 95] = [
    [0x00,0x00,0x00,0x00,0x00,0x00,0x00], // 32 ' '
    [0x04,0x04,0x04,0x04,0x04,0x00,0x04], // 33 '!'
    [0x0A,0x0A,0x0A,0x00,0x00,0x00,0x00], // 34 '"'
    [0x0A,0x0A,0x1F,0x0A,0x1F,0x0A,0x0A], // 35 '#'
    [0x04,0x0F,0x14,0x0E,0x05,0x1E,0x04], // 36 '$'
    [0x18,0x19,0x02,0x04,0x08,0x13,0x03], // 37 '%'
    [0x0C,0x12,0x14,0x08,0x15,0x12,0x0D], // 38 '&'
    [0x04,0x04,0x08,0x00,0x00,0x00,0x00], // 39 '''
    [0x02,0x04,0x08,0x08,0x08,0x04,0x02], // 40 '('
    [0x08,0x04,0x02,0x02,0x02,0x04,0x08], // 41 ')'
    [0x00,0x04,0x15,0x0E,0x15,0x04,0x00], // 42 '*'
    [0x00,0x04,0x04,0x1F,0x04,0x04,0x00], // 43 '+'
    [0x00,0x00,0x00,0x00,0x00,0x04,0x08], // 44 ','
    [0x00,0x00,0x00,0x1F,0x00,0x00,0x00], // 45 '-'
    [0x00,0x00,0x00,0x00,0x00,0x00,0x04], // 46 '.'
    [0x00,0x01,0x02,0x04,0x08,0x10,0x00], // 47 '/'
    [0x0E,0x11,0x13,0x15,0x19,0x11,0x0E], // 48 '0'
    [0x04,0x0C,0x04,0x04,0x04,0x04,0x0E], // 49 '1'
    [0x0E,0x11,0x01,0x02,0x04,0x08,0x1F], // 50 '2'
    [0x1F,0x02,0x04,0x02,0x01,0x11,0x0E], // 51 '3'
    [0x02,0x06,0x0A,0x12,0x1F,0x02,0x02], // 52 '4'
    [0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E], // 53 '5'
    [0x06,0x08,0x10,0x1E,0x11,0x11,0x0E], // 54 '6'
    [0x1F,0x01,0x02,0x04,0x08,0x08,0x08], // 55 '7'
    [0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E], // 56 '8'
    [0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C], // 57 '9'
    [0x00,0x00,0x04,0x00,0x00,0x04,0x00], // 58 ':'
    [0x00,0x00,0x04,0x00,0x00,0x04,0x08], // 59 ';'
    [0x02,0x04,0x08,0x10,0x08,0x04,0x02], // 60 '<'
    [0x00,0x00,0x1F,0x00,0x1F,0x00,0x00], // 61 '='
    [0x08,0x04,0x02,0x01,0x02,0x04,0x08], // 62 '>'
    [0x0E,0x11,0x01,0x02,0x04,0x00,0x04], // 63 '?'
    [0x0E,0x11,0x17,0x15,0x17,0x10,0x0E], // 64 '@'
    [0x0E,0x11,0x11,0x1F,0x11,0x11,0x11], // 65 'A'
    [0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E], // 66 'B'
    [0x0E,0x11,0x10,0x10,0x10,0x11,0x0E], // 67 'C'
    [0x1C,0x12,0x11,0x11,0x11,0x12,0x1C], // 68 'D'
    [0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F], // 69 'E'
    [0x1F,0x10,0x10,0x1E,0x10,0x10,0x10], // 70 'F'
    [0x0E,0x11,0x10,0x17,0x11,0x11,0x0F], // 71 'G'
    [0x11,0x11,0x11,0x1F,0x11,0x11,0x11], // 72 'H'
    [0x0E,0x04,0x04,0x04,0x04,0x04,0x0E], // 73 'I'
    [0x07,0x02,0x02,0x02,0x02,0x12,0x0C], // 74 'J'
    [0x11,0x12,0x14,0x18,0x14,0x12,0x11], // 75 'K'
    [0x10,0x10,0x10,0x10,0x10,0x10,0x1F], // 76 'L'
    [0x11,0x1B,0x15,0x15,0x11,0x11,0x11], // 77 'M'
    [0x11,0x11,0x19,0x15,0x13,0x11,0x11], // 78 'N'
    [0x0E,0x11,0x11,0x11,0x11,0x11,0x0E], // 79 'O'
    [0x1E,0x11,0x11,0x1E,0x10,0x10,0x10], // 80 'P'
    [0x0E,0x11,0x11,0x11,0x15,0x12,0x0D], // 81 'Q'
    [0x1E,0x11,0x11,0x1E,0x14,0x12,0x11], // 82 'R'
    [0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E], // 83 'S'
    [0x1F,0x04,0x04,0x04,0x04,0x04,0x04], // 84 'T'
    [0x11,0x11,0x11,0x11,0x11,0x11,0x0E], // 85 'U'
    [0x11,0x11,0x11,0x11,0x11,0x0A,0x04], // 86 'V'
    [0x11,0x11,0x11,0x15,0x15,0x1B,0x11], // 87 'W'
    [0x11,0x11,0x0A,0x04,0x0A,0x11,0x11], // 88 'X'
    [0x11,0x11,0x0A,0x04,0x04,0x04,0x04], // 89 'Y'
    [0x1F,0x01,0x02,0x04,0x08,0x10,0x1F], // 90 'Z'
    [0x0E,0x08,0x08,0x08,0x08,0x08,0x0E], // 91 '['
    [0x00,0x10,0x08,0x04,0x02,0x01,0x00], // 92 '\'
    [0x0E,0x02,0x02,0x02,0x02,0x02,0x0E], // 93 ']'
    [0x04,0x0A,0x11,0x00,0x00,0x00,0x00], // 94 '^'
    [0x00,0x00,0x00,0x00,0x00,0x00,0x1F], // 95 '_'
    [0x08,0x04,0x02,0x00,0x00,0x00,0x00], // 96 '`'
    [0x00,0x00,0x0E,0x01,0x0F,0x11,0x0F], // 97 'a'
    [0x10,0x10,0x16,0x19,0x11,0x11,0x1E], // 98 'b'
    [0x00,0x00,0x0E,0x10,0x10,0x11,0x0E], // 99 'c'
    [0x01,0x01,0x0D,0x13,0x11,0x11,0x0F], // 100 'd'
    [0x00,0x00,0x0E,0x11,0x1F,0x10,0x0E], // 101 'e'
    [0x06,0x09,0x08,0x1C,0x08,0x08,0x08], // 102 'f'
    [0x00,0x00,0x0F,0x11,0x0F,0x01,0x0E], // 103 'g'
    [0x10,0x10,0x16,0x19,0x11,0x11,0x11], // 104 'h'
    [0x04,0x00,0x0C,0x04,0x04,0x04,0x0E], // 105 'i'
    [0x02,0x00,0x06,0x02,0x02,0x12,0x0C], // 106 'j'
    [0x10,0x10,0x12,0x14,0x18,0x14,0x12], // 107 'k'
    [0x0C,0x04,0x04,0x04,0x04,0x04,0x0E], // 108 'l'
    [0x00,0x00,0x1A,0x15,0x15,0x11,0x11], // 109 'm'
    [0x00,0x00,0x16,0x19,0x11,0x11,0x11], // 110 'n'
    [0x00,0x00,0x0E,0x11,0x11,0x11,0x0E], // 111 'o'
    [0x00,0x00,0x1E,0x11,0x1E,0x10,0x10], // 112 'p'
    [0x00,0x00,0x0D,0x13,0x0F,0x01,0x01], // 113 'q'
    [0x00,0x00,0x16,0x19,0x10,0x10,0x10], // 114 'r'
    [0x00,0x00,0x0E,0x10,0x0E,0x01,0x1E], // 115 's'
    [0x08,0x08,0x1C,0x08,0x08,0x09,0x06], // 116 't'
    [0x00,0x00,0x11,0x11,0x11,0x13,0x0D], // 117 'u'
    [0x00,0x00,0x11,0x11,0x11,0x0A,0x04], // 118 'v'
    [0x00,0x00,0x11,0x11,0x15,0x15,0x0A], // 119 'w'
    [0x00,0x00,0x11,0x0A,0x04,0x0A,0x11], // 120 'x'
    [0x00,0x00,0x11,0x11,0x0F,0x01,0x0E], // 121 'y'
    [0x00,0x00,0x1F,0x02,0x04,0x08,0x1F], // 122 'z'
    [0x02,0x04,0x04,0x08,0x04,0x04,0x02], // 123 '{'
    [0x04,0x04,0x04,0x04,0x04,0x04,0x04], // 124 '|'
    [0x08,0x04,0x04,0x02,0x04,0x04,0x08], // 125 '}'
    [0x00,0x00,0x08,0x15,0x02,0x00,0x00], // 126 '~'
];

// ============================================================================
// Layout Constants
// ============================================================================

const IMG_W: u32 = 1200;
const IMG_H: u32 = 440;
const LEFT_PANEL_W: u32 = 200;
const RIGHT_PANEL_W: u32 = 200;
const FIELD_W: u32 = 800;
const FIELD_H: u32 = 400;
const LEGEND_H: u32 = 40;
const FIELD_PAD: u32 = 20;

// Default world coordinate bounds (used if no neurons provided)
const DEFAULT_X_MIN: f32 = -1.0;
const DEFAULT_X_MAX: f32 = 26.0;
const DEFAULT_Y_MIN: f32 = -1.0;
const DEFAULT_Y_MAX: f32 = 6.0;

// ============================================================================
// Data Structures
// ============================================================================

/// Metrics displayed in the side panels.
pub(crate) struct SnapshotMetrics {
    pub sim_time_us: u64,
    pub wall_clock_ms: f64,
    pub neuron_count: usize,
    pub synapse_count: usize,
    pub total_spikes: u64,
    pub total_events: u64,
    pub sensory_util: f32,
    pub inter_util: f32,
    pub motor_util: f32,
    pub learning_cycles: u32,
    pub total_strengthened: u32,
    pub total_weakened: u32,
    pub total_dormant: u32,
    pub mean_displacement: f32,
    pub region_count: usize,
}

/// Snapshot request sent from test thread to writer thread.
pub(crate) struct SnapshotRequest {
    pub label: String,
    pub seq: u32,
    pub neurons: Vec<SpatialNeuron>,
    pub metrics: SnapshotMetrics,
}

// ============================================================================
// Renderer
// ============================================================================

/// Computed world bounds with margin.
struct WorldBounds {
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
}

impl WorldBounds {
    fn from_neurons(neurons: &[SpatialNeuron]) -> Self {
        if neurons.is_empty() {
            return Self {
                x_min: DEFAULT_X_MIN,
                x_max: DEFAULT_X_MAX,
                y_min: DEFAULT_Y_MIN,
                y_max: DEFAULT_Y_MAX,
            };
        }
        let mut x_min = f32::MAX;
        let mut x_max = f32::MIN;
        let mut y_min = f32::MAX;
        let mut y_max = f32::MIN;
        for n in neurons {
            let x = n.soma.position[0];
            let y = n.soma.position[1];
            if x < x_min { x_min = x; }
            if x > x_max { x_max = x; }
            if y < y_min { y_min = y; }
            if y > y_max { y_max = y; }
        }
        // Add margin (10% of range, minimum 1.0)
        let x_margin = ((x_max - x_min) * 0.1).max(1.0);
        let y_margin = ((y_max - y_min) * 0.1).max(1.0);
        Self {
            x_min: x_min - x_margin,
            x_max: x_max + x_margin,
            y_min: y_min - y_margin,
            y_max: y_max + y_margin,
        }
    }
}

struct SnapshotRenderer {
    buf: Vec<u8>, // RGB8: IMG_W * IMG_H * 3
}

impl SnapshotRenderer {
    fn new() -> Self {
        Self {
            buf: vec![0u8; (IMG_W * IMG_H * 3) as usize],
        }
    }

    fn render(&mut self, req: &SnapshotRequest) -> &[u8] {
        let bounds = WorldBounds::from_neurons(&req.neurons);
        self.clear();
        self.draw_panel_backgrounds();
        self.draw_field_background();
        self.draw_grid_lines(&bounds);
        self.draw_neurons(&req.neurons, &bounds);
        self.draw_left_panel(&req.label, &req.metrics);
        self.draw_right_panel(&req.metrics);
        self.draw_legend();
        self.draw_borders();
        &self.buf
    }

    // --- Primitives ---

    #[inline]
    fn set_pixel(&mut self, x: u32, y: u32, color: [u8; 3]) {
        if x < IMG_W && y < IMG_H {
            let idx = ((y * IMG_W + x) * 3) as usize;
            self.buf[idx] = color[0];
            self.buf[idx + 1] = color[1];
            self.buf[idx + 2] = color[2];
        }
    }

    fn fill_rect(&mut self, x: u32, y: u32, w: u32, h: u32, color: [u8; 3]) {
        for dy in 0..h {
            for dx in 0..w {
                self.set_pixel(x + dx, y + dy, color);
            }
        }
    }

    fn draw_dot(&mut self, cx: u32, cy: u32, radius: u32, color: [u8; 3]) {
        let r = radius as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    let px = cx as i32 + dx;
                    let py = cy as i32 + dy;
                    if px >= 0 && py >= 0 {
                        self.set_pixel(px as u32, py as u32, color);
                    }
                }
            }
        }
    }

    fn draw_char(&mut self, x: u32, y: u32, ch: char, color: [u8; 3]) {
        let code = ch as u32;
        if code < 32 || code > 126 {
            return;
        }
        let glyph = &FONT_5X7[(code - 32) as usize];
        for (row, &bits) in glyph.iter().enumerate() {
            for col in 0..5u32 {
                if bits & (0x10 >> col) != 0 {
                    self.set_pixel(x + col, y + row as u32, color);
                }
            }
        }
    }

    fn draw_text(&mut self, x: u32, y: u32, text: &str, color: [u8; 3]) {
        for (i, ch) in text.chars().enumerate() {
            self.draw_char(x + i as u32 * CHAR_W, y, ch, color);
        }
    }

    fn draw_hline(&mut self, x: u32, y: u32, w: u32, color: [u8; 3]) {
        for dx in 0..w {
            self.set_pixel(x + dx, y, color);
        }
    }

    fn draw_vline(&mut self, x: u32, y: u32, h: u32, color: [u8; 3]) {
        for dy in 0..h {
            self.set_pixel(x, y + dy, color);
        }
    }

    // --- World-to-pixel mapping ---

    fn world_to_px(&self, wx: f32, wy: f32, b: &WorldBounds) -> (u32, u32) {
        let usable_w = FIELD_W - 2 * FIELD_PAD;
        let usable_h = FIELD_H - 2 * FIELD_PAD;
        let x_frac = (wx - b.x_min) / (b.x_max - b.x_min);
        let y_frac = (b.y_max - wy) / (b.y_max - b.y_min);
        let px = LEFT_PANEL_W + FIELD_PAD + (x_frac * usable_w as f32) as u32;
        let py = FIELD_PAD + (y_frac * usable_h as f32) as u32;
        (px.min(IMG_W - 1), py.min(IMG_H - 1))
    }

    // --- Composite drawing ---

    fn clear(&mut self) {
        for chunk in self.buf.chunks_exact_mut(3) {
            chunk[0] = Palette::BG[0];
            chunk[1] = Palette::BG[1];
            chunk[2] = Palette::BG[2];
        }
    }

    fn draw_panel_backgrounds(&mut self) {
        // Left panel
        self.fill_rect(0, 0, LEFT_PANEL_W, FIELD_H, Palette::PANEL_BG);
        // Right panel
        self.fill_rect(LEFT_PANEL_W + FIELD_W, 0, RIGHT_PANEL_W, FIELD_H, Palette::PANEL_BG);
        // Legend bar
        self.fill_rect(0, FIELD_H, IMG_W, LEGEND_H, Palette::PANEL_BG);
    }

    fn draw_field_background(&mut self) {
        self.fill_rect(LEFT_PANEL_W, 0, FIELD_W, FIELD_H, Palette::BG);
    }

    fn draw_grid_lines(&mut self, b: &WorldBounds) {
        let x_step = nice_step(b.x_max - b.x_min);
        let y_step = nice_step(b.y_max - b.y_min);

        // Vertical grid lines
        let mut wx = (b.x_min / x_step).ceil() * x_step;
        while wx <= b.x_max {
            let (px, _) = self.world_to_px(wx, 0.0, b);
            if px > LEFT_PANEL_W + FIELD_PAD && px < LEFT_PANEL_W + FIELD_W - FIELD_PAD {
                self.draw_vline(px, FIELD_PAD, FIELD_H - 2 * FIELD_PAD, Palette::GRID_LINE);
                let label = format_grid_label(wx);
                let lx = px.saturating_sub((label.len() as u32 * CHAR_W) / 2);
                self.draw_text(lx, FIELD_H - FIELD_PAD + 4, &label, Palette::TEXT_DIM);
            }
            wx += x_step;
        }
        // Horizontal grid lines
        let mut wy = (b.y_min / y_step).ceil() * y_step;
        while wy <= b.y_max {
            let (_, py) = self.world_to_px(0.0, wy, b);
            if py > FIELD_PAD && py < FIELD_H - FIELD_PAD {
                self.draw_hline(LEFT_PANEL_W + FIELD_PAD, py, FIELD_W - 2 * FIELD_PAD, Palette::GRID_LINE);
                let label = format_grid_label(wy);
                let lx = LEFT_PANEL_W + 4;
                self.draw_text(lx, py.saturating_sub(3), &label, Palette::TEXT_DIM);
            }
            wy += y_step;
        }
    }

    fn draw_neurons(&mut self, neurons: &[SpatialNeuron], b: &WorldBounds) {
        for n in neurons {
            let (px, py) = self.world_to_px(n.soma.position[0], n.soma.position[1], b);
            let fired = n.last_spike_us > 0;
            let (color, radius) = if n.nuclei.is_sensory() {
                if fired { (Palette::SENSORY_FIRED, 4) } else { (Palette::SENSORY_SILENT, 3) }
            } else if n.nuclei.is_motor() {
                if fired { (Palette::MOTOR_FIRED, 4) } else { (Palette::MOTOR_SILENT, 3) }
            } else if fired {
                (Palette::INTER_FIRED, 4)
            } else {
                (Palette::INTER_SILENT, 3)
            };
            self.draw_dot(px, py, radius, color);
        }
    }

    fn draw_left_panel(&mut self, label: &str, m: &SnapshotMetrics) {
        let x = 10u32;
        let mut y = 12u32;
        let step = CHAR_H + 1;

        // Title
        self.draw_text(x, y, label, Palette::LABEL);
        y += step;
        self.draw_hline(x, y, LEFT_PANEL_W - 20, Palette::BORDER);
        y += step;

        // Sim time
        self.draw_text(x, y, "Sim Time", Palette::TEXT_DIM);
        y += step;
        let sim_ms = m.sim_time_us as f64 / 1000.0;
        if sim_ms > 1000.0 {
            self.draw_text(x + 6, y, &format!("{:.1} s", sim_ms / 1000.0), Palette::TEXT);
        } else {
            self.draw_text(x + 6, y, &format!("{:.0} ms", sim_ms), Palette::TEXT);
        }
        y += step + 4;

        // Wall clock
        self.draw_text(x, y, "Wall Clock", Palette::TEXT_DIM);
        y += step;
        if m.wall_clock_ms > 1000.0 {
            self.draw_text(x + 6, y, &format!("{:.1} s", m.wall_clock_ms / 1000.0), Palette::TEXT);
        } else {
            self.draw_text(x + 6, y, &format!("{:.0} ms", m.wall_clock_ms), Palette::TEXT);
        }
        y += step + 4;

        // Network
        self.draw_text(x, y, "Network", Palette::TEXT_DIM);
        y += step;
        self.draw_text(x + 6, y, &format!("{} neurons", m.neuron_count), Palette::TEXT);
        y += step;
        self.draw_text(x + 6, y, &format!("{} synapses", m.synapse_count), Palette::TEXT);
        y += step + 4;

        // Activity
        self.draw_text(x, y, "Activity", Palette::TEXT_DIM);
        y += step;
        self.draw_text(x + 6, y, &format_count(m.total_spikes, "spikes"), Palette::TEXT);
        y += step;
        self.draw_text(x + 6, y, &format_count(m.total_events, "events"), Palette::TEXT);
    }

    fn draw_right_panel(&mut self, m: &SnapshotMetrics) {
        let x = LEFT_PANEL_W + FIELD_W + 10;
        let mut y = 12u32;
        let step = CHAR_H + 1;

        // Utilization
        self.draw_text(x, y, "Utilization", Palette::TEXT_DIM);
        y += step;
        self.draw_hline(x, y, RIGHT_PANEL_W - 20, Palette::BORDER);
        y += step;

        // Sensory bar
        self.draw_text(x, y, "Sensory", Palette::TEXT_DIM);
        y += step;
        self.draw_util_bar(x + 6, y, m.sensory_util, Palette::SENSORY_FIRED);
        y += step + 4;

        // Inter bar
        self.draw_text(x, y, "Inter", Palette::TEXT_DIM);
        y += step;
        self.draw_util_bar(x + 6, y, m.inter_util, Palette::INTER_FIRED);
        y += step + 4;

        // Motor bar
        self.draw_text(x, y, "Motor", Palette::TEXT_DIM);
        y += step;
        self.draw_util_bar(x + 6, y, m.motor_util, Palette::MOTOR_FIRED);
        y += step + 6;

        // Learning
        self.draw_text(x, y, "Learning", Palette::TEXT_DIM);
        y += step;
        self.draw_text(x + 6, y, &format!("{} cycles", m.learning_cycles), Palette::TEXT);
        y += step;
        self.draw_text(x + 6, y, &format!("+{} strengthen", m.total_strengthened), Palette::TEXT);
        y += step;
        self.draw_text(x + 6, y, &format!("-{} weakened", m.total_weakened), Palette::TEXT);
        y += step;
        self.draw_text(x + 6, y, &format!("{} dormant", m.total_dormant), Palette::TEXT);
        y += step + 4;

        // Spatial
        self.draw_text(x, y, "Spatial", Palette::TEXT_DIM);
        y += step;
        self.draw_text(x + 6, y, &format!("{:.3} displace", m.mean_displacement), Palette::TEXT);
        y += step;
        self.draw_text(x + 6, y, &format!("{} regions", m.region_count), Palette::TEXT);
    }

    fn draw_util_bar(&mut self, x: u32, y: u32, fraction: f32, color: [u8; 3]) {
        let bar_w = 100u32;
        let bar_h = 6u32;
        // Background
        self.fill_rect(x, y, bar_w, bar_h, Palette::GRID_LINE);
        // Filled portion
        let fill = (fraction.clamp(0.0, 1.0) * bar_w as f32) as u32;
        if fill > 0 {
            self.fill_rect(x, y, fill, bar_h, color);
        }
        // Percentage text
        self.draw_text(x + bar_w + 4, y.saturating_sub(1),
            &format!("{:.0}%", fraction * 100.0), Palette::TEXT);
    }

    fn draw_legend(&mut self) {
        let y = FIELD_H + 12;
        let entries: &[([u8; 3], &str)] = &[
            (Palette::SENSORY_FIRED, "Sensory (fired)"),
            (Palette::SENSORY_SILENT, "Sensory"),
            (Palette::INTER_FIRED, "Inter (fired)"),
            (Palette::INTER_SILENT, "Inter"),
            (Palette::MOTOR_FIRED, "Motor (fired)"),
            (Palette::MOTOR_SILENT, "Motor"),
        ];
        let spacing = IMG_W / entries.len() as u32;
        for (i, &(color, label)) in entries.iter().enumerate() {
            let x = spacing / 2 + i as u32 * spacing;
            self.draw_dot(x, y + 3, 5, color);
            self.draw_text(x + 10, y, label, Palette::TEXT);
        }
    }

    fn draw_borders(&mut self) {
        // Left panel right border
        self.draw_vline(LEFT_PANEL_W - 1, 0, FIELD_H, Palette::BORDER);
        // Right panel left border
        self.draw_vline(LEFT_PANEL_W + FIELD_W, 0, FIELD_H, Palette::BORDER);
        // Legend top border
        self.draw_hline(0, FIELD_H, IMG_W, Palette::BORDER);
    }
}

/// Pick a nice grid step for a given range (targeting ~5-6 lines).
fn nice_step(range: f32) -> f32 {
    let raw = range / 6.0;
    let magnitude = 10.0f32.powf(raw.log10().floor());
    let normalized = raw / magnitude;
    let step = if normalized < 1.5 {
        1.0
    } else if normalized < 3.5 {
        2.0
    } else if normalized < 7.5 {
        5.0
    } else {
        10.0
    };
    step * magnitude
}

fn format_grid_label(v: f32) -> String {
    if v.fract().abs() < 0.01 {
        format!("{}", v as i32)
    } else {
        format!("{:.1}", v)
    }
}

fn format_count(n: u64, suffix: &str) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M {}", n as f64 / 1_000_000.0, suffix)
    } else if n >= 1_000 {
        format!("{:.1}K {}", n as f64 / 1_000.0, suffix)
    } else {
        format!("{} {}", n, suffix)
    }
}

// ============================================================================
// Async Writer Thread
// ============================================================================

pub(crate) struct SnapshotWriter {
    sender: Option<mpsc::Sender<SnapshotRequest>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl SnapshotWriter {
    pub fn spawn(output_dir: PathBuf) -> Self {
        let (tx, rx) = mpsc::channel::<SnapshotRequest>();

        let handle = thread::spawn(move || {
            let mut renderer = SnapshotRenderer::new();

            while let Ok(req) = rx.recv() {
                let pixels = renderer.render(&req);
                let filename = format!("snapshot_{:03}_{}.png",
                    req.seq,
                    req.label.to_lowercase().replace(' ', "_"));
                let path = output_dir.join(&filename);

                if let Err(e) = image::save_buffer(
                    &path,
                    pixels,
                    IMG_W,
                    IMG_H,
                    image::ColorType::Rgb8,
                ) {
                    eprintln!("snapshot: failed to save {}: {}", path.display(), e);
                } else {
                    println!("    [snapshot] Saved {}", path.display());
                }
            }
        });

        Self {
            sender: Some(tx),
            handle: Some(handle),
        }
    }

    pub fn queue(&self, request: SnapshotRequest) {
        if let Some(ref tx) = self.sender {
            let _ = tx.send(request);
        }
    }

    /// Drop sender to signal thread exit, then join with 10s timeout.
    pub fn finish(mut self) {
        self.sender.take(); // drop sender → recv loop exits
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub(crate) fn snapshot_output_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_output")
        .join("snapshots");
    std::fs::create_dir_all(&dir).expect("failed to create snapshot output dir");
    dir
}
