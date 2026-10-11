#!/usr/bin/env python3
"""Original noise-capable service with an additional scenario-row API.

Uses the original service's CLI options and endpoints without modifying its file.
POST /scenario/row {"row": 0, "frozen": true} selects and holds row 0.
POST /scenario/row {"row": 0, "frozen": false} restarts the sweep at row 0.
The selected row remains active for one full scenario period before advancing.
Freeze holds the selected row; unfreeze resumes from it.
Powerflow results update asynchronously on the next completed PF calculation.
"""
import threading
import time

from fastapi import HTTPException
from pydantic import BaseModel, Field, StrictInt

import realtime_powerflow_service_with_noise as original

_base_create_app = original.create_app


class RowSelection(BaseModel):
    row: StrictInt = Field(ge=0)
    frozen: bool = True


class RowSelectablePowerflow(original.RealTimePowerflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scenario_condition = threading.Condition(self.lock)
        self._next_scenario_at = 0.0
        self.set_scenario_row(0, frozen=True)

    def _apply_row_locked(self, row):
        self.current_S_targets = {
            bus: complex(float(self.scenarios_df.iloc[row][f'P_node{j}']) / -1000,
                         float(self.scenarios_df.iloc[row][f'Q_node{j}']) / -1000)
            for j, bus in enumerate(original.DESIRED_SCENARIO_BUS_ORDER, start=1)
        }
        self.current_row_idx = row

    def set_scenario_row(self, row, frozen=True):
        if not 0 <= row < len(self.scenarios_df):
            raise ValueError(f'row must be between 0 and {len(self.scenarios_df)-1}')
        with self._scenario_condition:
            self._apply_row_locked(row)
            self.scenario_frozen = frozen
            self._next_scenario_at = time.monotonic() + 1 / max(1e-6, self.scenario_rate_hz)
            self._scenario_condition.notify_all()
            return {'ok': True, 'row_idx': row, 'frozen': frozen, 'n_rows': len(self.scenarios_df)}

    def set_scenario_frozen(self, frozen):
        with self._scenario_condition:
            self.scenario_frozen = bool(frozen)
            self._next_scenario_at = time.monotonic() + 1 / max(1e-6, self.scenario_rate_hz)
            self._scenario_condition.notify_all()

    def toggle_scenario_frozen(self):
        with self._scenario_condition:
            self.set_scenario_frozen(not self.scenario_frozen)
            return self.scenario_frozen

    def _scenario_loop(self):
        with self._scenario_condition:
            while not self._stop.is_set():
                remaining = self._next_scenario_at - time.monotonic()
                if self.scenario_frozen or remaining > 0:
                    self._scenario_condition.wait(timeout=0.1 if self.scenario_frozen else min(remaining, 0.1))
                    continue
                self._apply_row_locked((self.current_row_idx + 1) % len(self.scenarios_df))
                self._next_scenario_at = time.monotonic() + 1 / max(1e-6, self.scenario_rate_hz)


def create_app(rtp):
    app = _base_create_app(rtp)

    @app.post('/scenario/row')
    def select_row(selection: RowSelection):
        try:
            return rtp.set_scenario_row(selection.row, selection.frozen)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


if __name__ == '__main__':
    # Reuse the original CLI in this process only; its source remains untouched.
    original.RealTimePowerflow = RowSelectablePowerflow
    original.create_app = create_app
    original.main()
