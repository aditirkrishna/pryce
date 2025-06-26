# yaml_to_mlir.py: Convert symbolic risk contract YAML to MLIR DSL
import yaml
import sys

def yaml_to_mlir(yaml_path, mlir_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    contract = data['contract']
    risk = data['risk']
    mlir_lines = []
    mlir_lines.append(f'contract.call @{contract["name"]} {{')
    for k, v in contract.items():
        if k != 'name':
            if isinstance(v, str):
                mlir_lines.append(f'  {k}:  "{v}"')
            else:
                mlir_lines.append(f'  {k}:  {v}')
    mlir_lines.append('}\n')
    mlir_lines.append('risk.compute {')
    for greek in risk.get('greeks', []):
        mlir_lines.append(f'  greek:    "{greek}"')
    for scenario in risk.get('scenarios', []):
        for factor, shift in scenario.items():
            sign = '+' if float(shift) > 0 else ''
            mlir_lines.append(f'  scenario: {sign}{shift} {factor}')
    mlir_lines.append('}\n')
    with open(mlir_path, 'w') as f:
        f.write('\n'.join(mlir_lines))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python yaml_to_mlir.py contract.yaml risk_test.mlir')
        sys.exit(1)
    yaml_to_mlir(sys.argv[1], sys.argv[2])
