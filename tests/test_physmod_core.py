import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physmod.core import (  # noqa: E402
    estimate_modes,
    ks_pluck,
    measure_f0,
    modal_strike,
    parse_midi,
    render_midi,
    write_midi,
)


class PhysicalModelCoreTests(unittest.TestCase):
    def test_pluck_pitch_stays_within_five_cents(self):
        sample_rate = 4_000
        for target_hz in (110.0, 220.0, 329.63):
            with self.subTest(target_hz=target_hz):
                signal = ks_pluck(target_hz, 0.4, fs=sample_rate, velocity=0.9)
                measured_hz = measure_f0(
                    signal, fs=sample_rate, fmin=70, fmax=500
                )
                cents = 1_200 * np.log2(measured_hz / target_hz)
                self.assertLessEqual(abs(cents), 5.0)

    def test_mode_estimator_recovers_synthetic_fundamental_and_partial(self):
        sample_rate = 4_000
        fundamental_hz = 196.0
        signal = modal_strike(
            fundamental_hz, 0.5, fs=sample_rate, velocity=0.85
        )

        estimates = estimate_modes(
            signal, fs=sample_rate, n_modes=2, fmin=60
        )

        self.assertEqual(len(estimates), 2)
        expected = (fundamental_hz, fundamental_hz * 3.984)
        for (measured_hz, _tau, _amplitude), expected_hz in zip(estimates, expected):
            with self.subTest(expected_hz=expected_hz):
                self.assertAlmostEqual(measured_hz / expected_hz, 1.0, delta=0.01)

    def test_midi_round_trip_and_render_produce_finite_audio(self):
        events = [
            (0, "on", 69, 96, 0),
            (0, "on", 72, 80, 1),
            (480, "off", 69, 0, 0),
            (480, "off", 72, 0, 1),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            midi_path = Path(temp_dir) / "note.mid"
            write_midi(midi_path, events, tpq=480, bpm=120)
            parsed = parse_midi(midi_path)

        self.assertEqual(len(parsed), 4)
        self.assertEqual(parsed[0][1:], ("on", 69, 96, 0))
        self.assertEqual(parsed[1][1:], ("on", 72, 80, 1))
        self.assertAlmostEqual(parsed[0][0], 0.0)
        self.assertEqual(parsed[2][1:], ("off", 69, 0, 0))
        self.assertEqual(parsed[3][1:], ("off", 72, 0, 1))
        self.assertAlmostEqual(parsed[2][0], 0.5)
        self.assertAlmostEqual(parsed[3][0], 0.5)

        audio = render_midi(parsed, fs=4_000)
        self.assertEqual(audio.ndim, 1)
        self.assertTrue(np.isfinite(audio).all())
        self.assertGreater(np.max(np.abs(audio)), 0.0)


if __name__ == "__main__":
    unittest.main()
