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

from bentoml.artifact.fastai_model_artifact import _import_torch_module
from bentoml.artifact.fastai_model_artifact.fastai_utils import grab_idx, one_item


class FastaiModelForServing:
    def __init__(self, model):
        self.model = model
        self.torch = _import_torch_module()

    def predict(
        self, item, return_x=False, batch_first=True, with_dropout=False, **kwargs
    ):
        batch = one_item(item)
        res = self.pred_batch(batch=batch, with_dropout=with_dropout)
        raw_pred, x = grab_idx(res, 0, batch_first), batch[0]
        x = ds.x.reconstruct(grab_idx(x, 0))
        y = ds.y.reconstruct(pred, x) if has_arg(ds.y.reconstruct, 'x') else ds.y.reconstruct(pred)
        return (x, y, raw_pred, raw_pred)

        # batch = self.data.one_item(item)
        # res = self.pred_batch(batch=batch, with_dropout=with_dropout)
        # raw_pred, x = grab_idx(res, 0, batch_first=batch_first), batch[0]
        # norm = getattr(self.data, 'norm', False)
        # if norm:
        #     x = self.data.denorm(x)
        #     if norm.keywords.get('do_y', False): raw_pred = self.data.denorm(raw_pred)
        # ds = self.data.single_ds
        # pred = ds.y.analyze_pred(raw_pred, **kwargs)
        # x = ds.x.reconstruct(grab_idx(x, 0))
        # y = ds.y.reconstruct(pred, x) if has_arg(ds.y.reconstruct,
        #                                          'x') else ds.y.reconstruct(pred)
        # return (x, y, pred, raw_pred) if return_x else (y, pred, raw_pred)

    def pred_batch(
        self, ds_type, batch=None, reconstruct=False, with_dropout=False, activ=None
    ):
        xb, yb = batch
        with self.torch.no_grad():
            if not with_dropout:
                preds = ''
            else:
                preds = ''
            res = activ(preds[0])
        return res

        # if batch is not None:
        #     xb, yb = batch
        # else:
        #     xb, yb = self.data.one_batch(ds_type, detach=False, denorm=False)
        # cb_handler = CallbackHandler(self.callbacks)
        # xb, yb = cb_handler.on_batch_begin(xb, yb, train=False)
        # activ = ifnone(activ, _loss_func2activ(self.loss_func))
        # with torch.no_grad():
        #     if not with_dropout:
        #         preds = loss_batch(self.model.eval(), xb, yb, cb_handler=cb_handler)
        #     else:
        #         preds = loss_batch(self.model.eval().apply(self.apply_dropout), xb, yb,
        #                            cb_handler=cb_handler)
        #     res = activ(preds[0])
        # if not reconstruct: return res
