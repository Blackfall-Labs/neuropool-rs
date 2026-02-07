//! External interface — how spatial neurons connect to the outside world.
//!
//! Interface is pure data with a `kind: u8` field. The executor interprets
//! the kind and dispatches to registered handlers. This is extensible:
//! users can register custom handlers for kind values 200+.
//!
//! ## Kind Conventions
//!
//! | Range | Category |
//! |-------|----------|
//! | 0 | None (internal neuron) |
//! | 1-49 | Sensory variants |
//! | 50-99 | Motor variants |
//! | 100-149 | Memory variants |
//! | 150-199 | Chemical/modulator variants |
//! | 200+ | User-defined |
//!
//! ## Energy Gates
//!
//! Interfaces use energy thresholds to gate behavior:
//! - Below `low_ceiling` → low-energy action (e.g., read)
//! - Above `high_floor` → high-energy action (e.g., write)
//! - Between → neutral action (e.g., exclude)

/// External interface — pure data, no enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interface {
    /// Interface type ID (0 = none, meaning defined by executor)
    pub kind: u8,
    /// Target ID (channel, bank, chemical field — meaning depends on kind)
    pub target: u16,
    /// Modality (0 = none, 1 = auditory, 2 = visual, etc.)
    pub modality: u8,
    /// Energy gates for behavior
    pub gates: EnergyGates,
}

/// Energy thresholds that gate behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnergyGates {
    /// Below this → low-energy action (e.g., read)
    pub low_ceiling: i16,
    /// Above this → high-energy action (e.g., write)
    pub high_floor: i16,
    // Between low_ceiling and high_floor = neutral action
}

/// Action determined by energy level relative to gates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterfaceAction {
    /// Energy below low_ceiling (e.g., memory read, passive sensing)
    Low,
    /// Energy between low_ceiling and high_floor (e.g., exclude, neutral)
    Neutral,
    /// Energy above high_floor (e.g., memory write, motor output)
    High,
}

impl EnergyGates {
    /// Default gates: neutral zone from -1000 to +1000.
    pub const fn default() -> Self {
        Self {
            low_ceiling: -1000,
            high_floor: 1000,
        }
    }

    /// Sensory default: low threshold for input reception.
    pub const fn sensory_default() -> Self {
        Self {
            low_ceiling: -500,
            high_floor: 2000,
        }
    }

    /// Motor default: high threshold for output.
    pub const fn motor_default() -> Self {
        Self {
            low_ceiling: -2000,
            high_floor: 500,
        }
    }

    /// Memory default: low=read, high=write, neutral=exclude.
    pub const fn memory_default() -> Self {
        Self {
            low_ceiling: -5000,
            high_floor: 5000,
        }
    }

    /// Determine action from energy level.
    #[inline]
    pub const fn action(&self, energy: i16) -> InterfaceAction {
        if energy < self.low_ceiling {
            InterfaceAction::Low
        } else if energy > self.high_floor {
            InterfaceAction::High
        } else {
            InterfaceAction::Neutral
        }
    }
}

impl Default for EnergyGates {
    fn default() -> Self {
        Self::default()
    }
}

impl Interface {
    // Kind constants (conventions, not constraints)
    pub const KIND_NONE: u8 = 0;
    pub const KIND_SENSORY_BASE: u8 = 1;
    pub const KIND_MOTOR_BASE: u8 = 50;
    pub const KIND_MEMORY_BASE: u8 = 100;
    pub const KIND_CHEMICAL_BASE: u8 = 150;
    pub const KIND_USER_BASE: u8 = 200;

    // Modality constants
    pub const MODALITY_NONE: u8 = 0;
    pub const MODALITY_AUDITORY: u8 = 1;
    pub const MODALITY_VISUAL: u8 = 2;
    pub const MODALITY_TACTILE: u8 = 3;
    pub const MODALITY_CHEMICAL: u8 = 4;

    /// No external interface (internal neuron).
    #[inline]
    pub const fn none() -> Self {
        Self {
            kind: Self::KIND_NONE,
            target: 0,
            modality: Self::MODALITY_NONE,
            gates: EnergyGates::default(),
        }
    }

    /// Sensory interface (receives external input).
    #[inline]
    pub const fn sensory(channel: u16, modality: u8) -> Self {
        Self {
            kind: Self::KIND_SENSORY_BASE,
            target: channel,
            modality,
            gates: EnergyGates::sensory_default(),
        }
    }

    /// Auditory sensory interface.
    #[inline]
    pub const fn auditory(channel: u16) -> Self {
        Self::sensory(channel, Self::MODALITY_AUDITORY)
    }

    /// Visual sensory interface.
    #[inline]
    pub const fn visual(channel: u16) -> Self {
        Self::sensory(channel, Self::MODALITY_VISUAL)
    }

    /// Motor interface (outputs to actuator).
    #[inline]
    pub const fn motor(channel: u16, modality: u8) -> Self {
        Self {
            kind: Self::KIND_MOTOR_BASE,
            target: channel,
            modality,
            gates: EnergyGates::motor_default(),
        }
    }

    /// Memory interface (accesses databank).
    #[inline]
    pub const fn memory(bank_id: u16) -> Self {
        Self {
            kind: Self::KIND_MEMORY_BASE,
            target: bank_id,
            modality: Self::MODALITY_NONE,
            gates: EnergyGates::memory_default(),
        }
    }

    /// Chemical/modulator interface.
    #[inline]
    pub const fn chemical(chemical_id: u8) -> Self {
        Self {
            kind: Self::KIND_CHEMICAL_BASE,
            target: chemical_id as u16,
            modality: Self::MODALITY_NONE,
            gates: EnergyGates::default(),
        }
    }

    /// User-defined interface.
    #[inline]
    pub const fn user(kind_offset: u8, target: u16) -> Self {
        Self {
            kind: Self::KIND_USER_BASE.saturating_add(kind_offset),
            target,
            modality: Self::MODALITY_NONE,
            gates: EnergyGates::default(),
        }
    }

    /// Is this an external interface (kind != 0)?
    #[inline]
    pub const fn is_external(&self) -> bool {
        self.kind != Self::KIND_NONE
    }

    /// Is this a sensory interface?
    #[inline]
    pub const fn is_sensory(&self) -> bool {
        self.kind >= Self::KIND_SENSORY_BASE && self.kind < Self::KIND_MOTOR_BASE
    }

    /// Is this a motor interface?
    #[inline]
    pub const fn is_motor(&self) -> bool {
        self.kind >= Self::KIND_MOTOR_BASE && self.kind < Self::KIND_MEMORY_BASE
    }

    /// Is this a memory interface?
    #[inline]
    pub const fn is_memory(&self) -> bool {
        self.kind >= Self::KIND_MEMORY_BASE && self.kind < Self::KIND_CHEMICAL_BASE
    }

    /// Is this a chemical/modulator interface?
    #[inline]
    pub const fn is_chemical(&self) -> bool {
        self.kind >= Self::KIND_CHEMICAL_BASE && self.kind < Self::KIND_USER_BASE
    }

    /// Determine action from current energy level.
    #[inline]
    pub const fn action(&self, energy: i16) -> InterfaceAction {
        self.gates.action(energy)
    }
}

impl Default for Interface {
    fn default() -> Self {
        Self::none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interface_categories() {
        assert!(!Interface::none().is_external());
        assert!(Interface::sensory(0, 1).is_sensory());
        assert!(Interface::motor(0, 1).is_motor());
        assert!(Interface::memory(0).is_memory());
        assert!(Interface::chemical(0).is_chemical());
    }

    #[test]
    fn test_energy_gates() {
        let gates = EnergyGates::memory_default();
        assert_eq!(gates.action(-6000), InterfaceAction::Low);
        assert_eq!(gates.action(0), InterfaceAction::Neutral);
        assert_eq!(gates.action(6000), InterfaceAction::High);
    }

    #[test]
    fn test_interface_action() {
        let mem = Interface::memory(0);
        assert_eq!(mem.action(-6000), InterfaceAction::Low); // read
        assert_eq!(mem.action(0), InterfaceAction::Neutral); // exclude
        assert_eq!(mem.action(6000), InterfaceAction::High); // write
    }
}
