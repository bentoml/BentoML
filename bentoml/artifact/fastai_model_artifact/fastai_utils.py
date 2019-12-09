# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def is_listy(x):
    return isinstance(x, (tuple, list))


def grab_idx(x, i, batch_first):
    if batch_first:
        return [o[i].cpu() for o in x] if is_listy(x) else x[i].cpu()
    else:
        return [o[:, i].cpu() for o in x] if is_listy(x) else x[:, i].cpu()


def one_item(self, item, detach=False, denorm=False, cpu=False):
    "Get `item` into a batch. Optionally `detach` and `denorm`."
    ds = self.single_ds
    with ds.set_item(item):
        return self.one_batch(ds_type=DatasetType.Single, detach=detach,
                              denorm=denorm, cpu=cpu)


