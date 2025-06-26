// Symbolic Risk DSL Example: Digital Option with Scenario and Risk Hooks

contract.call @digital_call {
  asset:  "SPX"
  type:   "digital"
  strike: 4200.0
  payout: 10.0
  expiry: 0.5
}

risk.compute {
  greek:    "delta"
  greek:    "gamma"
  greek:    "vega"
  scenario: +0.01 spot
  scenario: -0.01 vol
}

// This file expresses both contract and risk logic in a symbolic, human-readable way.
// The pipeline will lower this to MC pricing, scenario grid, and adjoint risk kernels.
