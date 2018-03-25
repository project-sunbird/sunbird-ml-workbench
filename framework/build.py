# -*- coding: utf-8 -*-
#
# Copyright (C) 2016  EkStep Foundation
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from pybuilder.core import init, use_plugin

use_plugin("python.core")
# use_plugin("python.unittest")
# use_plugin("python.coverage")
use_plugin("python.install_dependencies")
use_plugin("python.distutils")
use_plugin('copy_resources')
use_plugin("exec")

default_task = "publish"

name = "mlworkbench"
version = "0.0.1"
license = "AGPL-3.0-or-later"

@init
def initialize(project):
    project.depends_on_requirements("requirements.txt")
    project.build_depends_on('mockito')
    project.get_property('copy_resources_glob').append('LICENCE')
    #project.set_property('publish_command', 'cp target/dist/mlworkbench-0.0.1/dist/mlworkbench-0.0.1.tar.gz ../dist')
