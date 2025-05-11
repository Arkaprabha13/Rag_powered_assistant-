import pkg_resources

installed_packages = [(d.project_name, d.version) for d in pkg_resources.working_set]
installed_packages_sorted = sorted(installed_packages)
for package, version in installed_packages_sorted:
    print(f"{package}=={version}")
