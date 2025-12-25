# attribute_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


class AttributeHead(nn.Module):
    def __init__(self, emb_dim, num_attrs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_attrs)
        )

    def forward(self, x):
        return self.net(x)


class AttributePredictor:
    """
    Attribute predictor using YOLO embedding + trained attribute head
    """

    def __init__(
        self,
        yolo_model,
        attr_ckpt_path,
        device="cuda",
        attr_thresh=0.01,
        hook_layer=22
    ):
        self.device = device
        self.attr_thresh = attr_thresh
        self.yolo = yolo_model

        # ---------- load checkpoint ----------
        ckpt = torch.load(attr_ckpt_path, map_location=device)
        self.idx2attr = ckpt["idx2attr"]
        self.emb_dim = ckpt["emb_dim"]

        self.attr_head = AttributeHead(
            emb_dim=self.emb_dim,
            num_attrs=len(self.idx2attr)
        ).to(device)

        self.attr_head.load_state_dict(ckpt["state_dict"])
        self.attr_head.eval()

        # ---------- embedding hook ----------
        self._embeddings = []

        def hook_fn(module, inp, out):
            self._embeddings.append(out)

        self.yolo.model.model[hook_layer].register_forward_hook(hook_fn)

        # ---------- image transform ----------
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def _extract_embedding(self, img_tensor):
        """
        img_tensor: [1, 3, H, W]
        """
        self._embeddings.clear()
        with torch.no_grad():
            self.yolo.model(img_tensor)

        feat = self._embeddings[0]              # [1, C, H, W]
        pooled = F.adaptive_avg_pool2d(feat, (1, 1))
        return pooled.view(pooled.size(0), -1)  # [1, C]

    def predict_from_crop(self, crop_bgr):
        """
        crop_bgr: numpy BGR image
        return: list[str] attributes
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return []

        crop_rgb = crop_bgr[..., ::-1]
        crop_pil = Image.fromarray(crop_rgb)

        img_t = self.transform(crop_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self._extract_embedding(img_t)
            probs = torch.sigmoid(self.attr_head(emb)).squeeze(0)

        attrs = [
            self.idx2attr[i]
            for i, p in enumerate(probs)
            if p.item() > self.attr_thresh
        ]

        return attrs
