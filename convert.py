import yaml

# Load the exported environment file
with open('environment.yml', 'r') as file:
    env_data = yaml.safe_load(file)

# Extract the dependencies
dependencies = env_data['dependencies']

# Write dependencies to requirements.txt
with open('requirements.txt', 'w') as file:
    for dep in dependencies:
        if isinstance(dep, str):
            # Ignore dependencies from conda-forge and other conda channels
            if dep.startswith('pip') or dep.startswith('-c'):
                continue
            file.write(dep.split('=')[0] + '\n')
        elif isinstance(dep, dict):
            for key, value in dep.items():
                if key == 'pip':
                    for pip_dep in value:
                        file.write(pip_dep + '\n')
