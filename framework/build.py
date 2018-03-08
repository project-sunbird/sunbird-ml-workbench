# [TODO] compile the ML workbench
# [TODO] run tests
# [TODO] check for lints
# [TODO] packaging - framwork and core operators

from pybuilder.core import init, use_plugin

use_plugin("python.core")
# use_plugin("python.unittest")
use_plugin("python.coverage")
use_plugin("python.install_dependencies")
use_plugin("python.distutils")

default_task = "publish"

@init
def initialize(project):
    project.build_depends_on('mockito')