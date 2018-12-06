from pybuilder.core import init, use_plugin

use_plugin("python.core")
use_plugin("python.install_dependencies")
use_plugin("python.distutils")
use_plugin('copy_resources')
use_plugin("python.unittest")
# use_plugin("python.coverage")
use_plugin('python.flake8')
use_plugin("exec")
use_plugin('python.pycharm')
use_plugin('python.sphinx')
use_plugin('python.integrationtest')

default_task = "publish"

name = "daggit"
version = "0.5.0"
license = "MIT License"


@init
def initialize(project):
    project.plugin_depends_on("flake8", "~=3.5")
    project.depends_on_requirements("requirements.txt")
    project.build_depends_on('mockito')
    project.set_property_if_unset("filter_resources_target", "$dir_target")
    project.get_property('copy_resources_glob').append('LICENSE')
    project.set_property_if_unset("flake8_break_build", False)
    project.set_property_if_unset("flake8_max_line_length", 120)
    project.set_property_if_unset("flake8_include_patterns", None)
    project.set_property_if_unset("flake8_exclude_patterns", None)
    project.set_property_if_unset("flake8_include_test_sources", False)
    project.set_property_if_unset("flake8_include_scripts", True)
    project.set_property_if_unset("flake8_max_complexity", None)
    project.set_property_if_unset("flake8_verbose_output", False)

    project.set_property("sphinx_config_path", "docs/source/")
    project.set_property("sphinx_source_dir", "docs/source/")
    project.set_property("sphinx_output_dir", "docs/_build")
