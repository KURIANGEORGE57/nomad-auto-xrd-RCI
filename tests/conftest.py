#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import logging
import os
import shutil

import pytest
import structlog
from nomad.client import normalize_all, parse
from nomad.utils import structlogging
from structlog.testing import LogCapture

structlogging.ConsoleFormatter.short_format = True
setattr(logging, 'Formatter', structlogging.ConsoleFormatter)


@pytest.fixture(
    name='caplog',
    scope='function',
)
def fixture_caplog(request):
    """
    Extracts log messages from the logger and raises an assertion error if the specified
    log levels in the `request.param` are found.
    """
    caplog = LogCapture()
    processors = structlog.get_config()['processors']
    old_processors = processors.copy()

    try:
        processors.clear()
        processors.append(caplog)
        structlog.configure(processors=processors)
        yield caplog
        for record in caplog.entries:
            if record['log_level'] in request.param:
                raise AssertionError(
                    f"Log level '{record['log_level']}' found: {record}"
                )
    finally:
        processors.clear()
        processors.extend(old_processors)
        structlog.configure(processors=processors)


@pytest.fixture(
    name='parsed_measurement_archives',
    scope='function',
)
def fixture_parsed_measurement_archives(request, clean_up):
    """
    Sets up data for testing and cleans up after the test. The data file is parsed,
    returning an `EntryArchive` object. It contains a reference to the `.archive.json`
    file created by plugin parsers for the measurement data. Parsing this
    `.archive.json` file returns the `EntryArchive` object for the measurement data,
    which is finally yeilded to the test function.

    Parameters:
        request.param (list): List of relative file paths to the measurement data files.
    """
    clean_up_extensions = ['.archive.json', '.nxs', '.h5']

    measurement_archives = []
    for rel_file_path in request.param:
        parse(rel_file_path)
        rel_measurement_archive_path = os.path.join(
            rel_file_path.rsplit('.', 1)[0] + '.archive.json'
        )
        measurement_archive = parse(rel_measurement_archive_path)[0]
        normalize_all(measurement_archive)
        measurement_archives.append(measurement_archive)

    yield measurement_archives

    # Add files to the clean_up fixture params
    clean_up.param = [
        rel_file_path.rsplit('.', 1)[0] + ext
        for rel_file_path in request.param
        for ext in clean_up_extensions
    ]


@pytest.fixture(name='clean_up', scope='function')
def fixture_clean_up():
    """
    Fixture for tracking and cleaning up files after tests.
    """
    files_to_clean = []

    class CleanUp:
        @property
        def param(self):
            return files_to_clean

        @param.setter
        def param(self, value):
            if isinstance(value, str):
                files_to_clean.append(value)
            elif isinstance(value, list | tuple):
                files_to_clean.extend(value)
            else:
                raise TypeError(f'Expected str, list, or tuple, got {type(value)}')

        def track(self, path):
            """Register a file for cleanup"""
            files_to_clean.append(path)
            return path

    yield CleanUp()

    # Clean up registered files
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Warning: Failed to clean up {file_path}: {e}')


# ============================================================================
# ARCO-specific test configuration
# ============================================================================


def pytest_configure(config):
    """
    Configure pytest with custom markers and behavior for ARCO tests.
    """
    # Add ARCO-specific markers
    config.addinivalue_line(
        'markers', 'smoke: Fast smoke tests that run on every commit (~30s)'
    )
    config.addinivalue_line(
        'markers', 'slow: Slower integration tests (~5min)'
    )
    config.addinivalue_line(
        'markers', 'pipeline: Heavy pipeline tests requiring full dependencies'
    )
    config.addinivalue_line('markers', 'arco: ARCO-specific tests')


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection based on environment variables.

    - If RUN_PIPELINE_TESTS is not set, skip pipeline tests
    - If RUN_SMOKE_ONLY is set, only run smoke tests
    """
    import os

    run_pipeline = os.getenv('RUN_PIPELINE_TESTS', '').lower() in ('true', '1', 'yes')
    run_smoke_only = os.getenv('RUN_SMOKE_ONLY', '').lower() in ('true', '1', 'yes')

    skip_pipeline = pytest.mark.skip(reason='RUN_PIPELINE_TESTS not set')
    skip_non_smoke = pytest.mark.skip(reason='RUN_SMOKE_ONLY is set, skipping non-smoke tests')

    for item in items:
        # Skip pipeline tests unless explicitly enabled
        if 'pipeline' in item.keywords and not run_pipeline:
            item.add_marker(skip_pipeline)

        # If smoke-only mode, skip non-smoke tests
        if run_smoke_only and 'smoke' not in item.keywords:
            item.add_marker(skip_non_smoke)


@pytest.fixture(scope='session')
def arco_test_data():
    """
    Fixture providing common test data for ARCO tests.
    """
    import numpy as np

    np.random.seed(42)

    return {
        'periodic_signal': np.sin(2 * np.pi * 0.1 * np.arange(1024)),
        'noise_signal': np.random.randn(1024),
        'heptad_pattern': np.tile([1, 0, 0, 2, 0, 0, 1], 146),  # ~1024 points
        'two_theta': np.linspace(10, 80, 512),
    }
