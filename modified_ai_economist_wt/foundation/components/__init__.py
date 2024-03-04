# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from modified_ai_economist_wt.foundation.base.base_component import component_registry

from . import (
    continuous_double_auction,
    move,
    redistribution,
    vote_and_invest_component,
    teaching_and_build_component,
)

# Import files that add Component class(es) to component_registry
# ---------------------------------------------------------------