import pkg_resources
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Load required packages from file
with open("requirements.txt") as req_file:
    required = pkg_resources.parse_requirements(req_file)
    required_pkgs = {req.key: str(req.specifier) for req in required}

# Get installed packages
installed_pkgs = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

# Check and print status
print(f"\n{Style.BRIGHT}📦 Dependency Diagnostic\n")
for pkg, version_spec in required_pkgs.items():
    if pkg in installed_pkgs:
        if version_spec and not pkg_resources.Requirement.parse(f"{pkg}{version_spec}").specifier.contains(installed_pkgs[pkg]):
            print(f"{Fore.YELLOW}⚠️ {pkg} installed version {installed_pkgs[pkg]} does not match required {version_spec}")
        else:
            print(f"{Fore.GREEN}✅ {pkg} {installed_pkgs[pkg]} installed")
    else:
        print(f"{Fore.RED}❌ {pkg} is missing")

print(f"\n{Style.BRIGHT}Done. 🔍")
