from dataclasses import dataclass
from typing import override
import pygame
import sys
from abc import abstractmethod, ABCMeta
import json

import torch
from torch import nn
import torch.nn.functional as F
import math

### ==============================
### Helpers
### ==============================

def fade_color(color, factor):
    return [max(0, int(c * factor)) for c in color]


@dataclass
class Window:
    width: int
    height: int
    screen: pygame.Surface | None = None


@dataclass
class Game:
    ai: list
    ai_history: list[list]
    heatmap: pygame.Surface
    draw: pygame.draw
    window: Window
    clock: pygame.time.Clock


### ==============================
### Drawable Base
### ==============================

class Drawable(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def event_hook(cls, event: pygame.event.Event):
        pass

    @classmethod
    @abstractmethod
    def draw(cls, game: Game):
        pass


### ==============================
### Click Marker & Replay
### ==============================

class PointMarker(Drawable):
    def __init__(self):
        super().__init__()
        self.point_list: list[list] = []
        self.ai_point_list: list[float] = []  # flat [x0,y0,x1,y1,...]
        self.previous_clicks: list[list] = []
        self.previous_clicks_position: int = 0

    def add(self, point: list):
        self.push([30, (point[0], point[1]), [
            [[80, 120, 80], 20],
            [[160, 240, 160], 10]
        ]])

    def push(self, point: list):
        timer, pos, state = point
        x = pos[0]
        y = pos[1]
        self.point_list.append(point)
        self.ai_point_list.append(x)
        self.ai_point_list.append(y)

    @override
    def event_hook(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == pygame.BUTTON_LEFT:
                self.previous_clicks.append([event.pos[0], event.pos[1]])
                print(event.dict, event.pos)
                self.add(event.pos)

    @override
    def draw(self, game: Game):
        for circle in self.point_list[:]:
            timer, pos, state = circle
            game.draw.circle(game.window.screen, state[0][0], pos, 20)
            game.draw.circle(game.window.screen, state[1][0], pos, 10)
            circle[0] -= 1
            if circle[0] % 10 == 0:
                state[0][0] = fade_color(state[0][0], 0.7)
                state[1][0] = fade_color(state[1][0], 0.5)

            if circle[0] <= 0:
                self.point_list.remove(circle)


### ==============================
### Spatial Transformer Model
### ==============================
class ClickViT(nn.Module):
    def __init__(
        self,
        window_size: int,
        grid_size: tuple[int,int],
        chan_dim: int = 32,
        num_heads: int = 2,
        mlp_hidden: int = 128,
    ):
        super().__init__()

        self.window_size = window_size
        self.grid_H, self.grid_W = grid_size
        self.d_model = chan_dim

        assert chan_dim % num_heads == 0

        # ---- [T,X,Y] → [C,X,Y] ----
        self.temporal_proj = nn.Sequential(
            nn.Conv2d(window_size, chan_dim, 1),
            nn.BatchNorm2d(chan_dim),
            nn.GELU(),
        )

        # ---- positional embedding ----
        self.register_buffer(
            "pos2d",
            self._build_pos(self.grid_H, self.grid_W, chan_dim),
            persistent=False
        )

        # ---- first light transformer ----
        enc1 = nn.TransformerEncoderLayer(
            d_model=chan_dim,
            nhead=num_heads,
            dim_feedforward=chan_dim * 2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer1 = nn.TransformerEncoder(enc1, 1)

        # ---- big global brain ----
        self.global_mlp = nn.Sequential(
            nn.LayerNorm(chan_dim),
            nn.Linear(chan_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, chan_dim),
        )

        # projection to inject global context back
        self.inject = nn.Linear(chan_dim, chan_dim)

        # ---- second light transformer ----
        enc2 = nn.TransformerEncoderLayer(
            d_model=chan_dim,
            nhead=num_heads,
            dim_feedforward=chan_dim * 2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer2 = nn.TransformerEncoder(enc2, 1)

        # ---- global skips are gated ----
        self.skip1_gate = nn.Parameter(torch.tensor(0.2))
        self.skip2_gate = nn.Parameter(torch.tensor(0.2))

        self.final_norm = nn.LayerNorm(chan_dim)

        # ---- output head ----
        self.head = nn.Linear(chan_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # # ---- output head (beefy) ----
        # self.head = nn.Sequential(
        #     nn.LayerNorm(chan_dim),

        #     nn.Linear(chan_dim, chan_dim * 2),
        #     nn.GELU(),

        #     nn.Linear(chan_dim * 2, chan_dim),
        #     nn.GELU(),

        #     # residual stab
        #     nn.LayerNorm(chan_dim),

        #     nn.Linear(chan_dim, 1)
        # )

        # # start gentle — avoids insane spikes early learning
        # for m in self.head:
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight, gain=0.5)
        #         nn.init.zeros_(m.bias)


    # --------------------------------
    def _build_pos(self, H, W, dim):
        assert dim % 4 == 0
        half = dim // 2
        yy, xx = torch.meshgrid(
            torch.arange(H), torch.arange(W), indexing="ij"
        )

        def enc(n, d):
            pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
            pe = torch.zeros(n, d)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            return pe

        pe_x = enc(W, half).unsqueeze(0).expand(H, -1, -1)
        pe_y = enc(H, half).unsqueeze(1).expand(-1, W, -1)
        pe = torch.cat([pe_x, pe_y], dim=-1)
        return pe.view(1, H * W, dim)

    # --------------------------------
    def forward(self, grid_seq):
        """
        grid_seq: (B,T,H,W)
        """
        B,T,H,W = grid_seq.shape

        # ---- temporal collapse ----
        x = self.temporal_proj(grid_seq)            # (B,C,H,W)

        # flatten
        tokens0 = x.permute(0,2,3,1).reshape(B, H*W, self.d_model)
        tokens0 = tokens0 + self.pos2d             # positional add

        # ---- Transformer 1 ----
        x1 = self.transformer1(tokens0)

        # ---- skip 1 ----
        x1 = x1 + self.skip1_gate * tokens0

        # ---- global summary ----
        g = x1.mean(dim=1)
        g2 = self.global_mlp(g)

        # inject global context everywhere
        inj = self.inject(g2).unsqueeze(1)
        z = x1 + inj

        # ---- Transformer 2 ----
        z2 = self.transformer2(z)

        # ---- skip 2 ----
        z2 = z2 + self.skip2_gate * z

        out = self.final_norm(z2)

        logits = self.head(out)          # (B,HW,1)
        logits = logits.view(B,1,H,W)
        return logits


### ==============================
### Predictor Wrapper (Batched)
### ==============================

class NNPredictor(nn.Module):
    def __init__(
        self,
        game,
        window_size: int = 6,
        batch_size: int = 32,
        max_history_examples: int = 128,
    ):
        super().__init__()
        self.window_size = window_size
        self.inputLength = self.window_size * 2  # flat length per example
        self.game = game
        self.clicks = 0  # how many coords we've already trained on
        self.batch_size = batch_size
        self.max_history_examples = max_history_examples

        # device
        self.device = (
            torch.accelerator.current_accelerator().type
            if hasattr(torch, "accelerator") and torch.accelerator.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print("Device:", self.device)

        self.register_buffer(
            "screen_size",
            torch.as_tensor(
                [game.window.width - 1, game.window.height - 1],
                dtype=torch.float32,
            ),
        )

        # grid resolution for model (doesn't have to equal screen pixels)
        self.grid_H = 32
        self.grid_W = 64

        self.model = ClickViT(
            window_size=self.window_size,
            grid_size=(self.grid_H, self.grid_W),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.losses: list[torch.Tensor] = []

        self.to(self.device)

    # ------------------------------
    # Build batch of (features, labels) from click history
    # ------------------------------
    def _build_batch(self, flat_clicks):
        """
        flat_clicks: [x0, y0, x1, y1, ...] in pixel coordinates

        We build examples for each t:
          - features_t: previous `window_size` clicks (flattened)
          - label_t:    click at time t (pixel coords)
        Then we pick up to `batch_size` of those (most recent).
        """
        num_coords = len(flat_clicks)
        num_clicks = num_coords // 2

        if num_clicks <= self.window_size:
            return None  # not enough data yet

        # all possible label indices t
        t_all = list(range(self.window_size, num_clicks))  # each t = label index

        # limit how far back we look for training
        if len(t_all) > self.max_history_examples:
            t_all = t_all[-self.max_history_examples:]

        # always include the latest index
        t_last = t_all[-1]

        if len(t_all) <= self.batch_size:
            chosen = t_all
        else:
            # choose batch_size-1 spaced indices + t_last
            needed = self.batch_size - 1
            pool = t_all[:-1]
            if len(pool) <= needed:
                chosen_except = pool
            else:
                step = len(pool) / needed
                chosen_except = [pool[int(i * step)] for i in range(needed)]
            chosen = chosen_except + [t_last]

        B = len(chosen)
        features = torch.empty(
            B, self.inputLength, device=self.device, dtype=torch.float32
        )
        labels = torch.empty(B, 2, device=self.device, dtype=torch.float32)

        for i, t in enumerate(chosen):
            # window of previous window_size clicks
            start = 2 * (t - self.window_size)
            end = 2 * t
            features[i, :] = torch.tensor(
                flat_clicks[start:end], dtype=torch.float32, device=self.device
            )

            # label is click at index t
            labels[i, :] = torch.tensor(
                flat_clicks[2 * t : 2 * t + 2],
                dtype=torch.float32,
                device=self.device,
            )

        latest_index = B - 1
        return features, labels, latest_index

    # ------------------------------
    # Build (B,T,H,W) grid from feature batch
    # ------------------------------
    def _build_grid_seq(self, features):
        """
        features: (B, inputLength) in pixel coords [x0,y0,...,x_{T-1},y_{T-1}]
        Returns:
           grid_seq: (B, T, H, W)
             - for each example, each timestep t has a 1 at the grid cell
               corresponding to that click, 0 elsewhere.
        """
        B, L = features.shape
        assert L == self.inputLength
        T = self.window_size
        H, W = self.grid_H, self.grid_W

        coords = features.view(B, T, 2)  # (B,T,2) in pixels

        # normalize to [0,1], then to grid indices
        xs = coords[..., 0] / self.screen_size[0] * (W - 1)
        ys = coords[..., 1] / self.screen_size[1] * (H - 1)

        x_idx = torch.clamp(xs.round().long(), 0, W - 1)
        y_idx = torch.clamp(ys.round().long(), 0, H - 1)

        grid_seq = torch.zeros(
            B, T, H, W, device=self.device, dtype=torch.float32
        )

        b_idx = torch.arange(B, device=self.device).view(B, 1).expand(B, T)
        t_idx = torch.arange(T, device=self.device).view(1, T).expand(B, T)

        grid_seq[b_idx, t_idx, y_idx, x_idx] = 1.0
        return grid_seq

    # ------------------------------
    # Forward on a batch of features
    # ------------------------------
    def forward_batch(self, features):
        grid_seq = self._build_grid_seq(features)      # (B,T,H,W)
        return self.model(grid_seq)                    # (B,1,H,W)

    # ------------------------------
    # Pixel-space expected distance loss (batched)
    # ------------------------------
    def _loss(self, logits, labels):
        """
        logits: (B,1,H,W)
        labels: (B,2) pixel coords [x_px, y_px]
        """
        B, C, H, W = logits.shape
        probs = torch.softmax(logits.view(B, -1), dim=-1).view(B, 1, H, W)

        xs = torch.arange(W, device=logits.device).view(1, 1, 1, W)
        ys = torch.arange(H, device=logits.device).view(1, 1, H, 1)

        x_label = labels[:, 0].view(B, 1, 1, 1)
        y_label = labels[:, 1].view(B, 1, 1, 1)

        dx = xs - x_label
        dy = ys - y_label
        dist = torch.sqrt(dx * dx + dy * dy)  # (B,1,H,W)

        loss_per_sample = (probs * dist).sum(dim=(1, 2, 3))
        loss = loss_per_sample.mean()
        return loss, probs

    # ------------------------------
    # Decode one sample from probs for UI
    # ------------------------------
    def _argmax_for_example(self, probs, sample_idx):
        """
        probs: (B,1,H,W)
        sample_idx: which example in batch to decode for UI
        """
        p = probs[sample_idx : sample_idx + 1]  # (1,1,H,W)
        _, _, H, W = p.shape
        idx = p.view(-1).argmax().item()
        y = idx // W
        x = idx % W

        X = int(round(x * self.screen_size[0].item() / (W - 1)))
        Y = int(round(y * self.screen_size[1].item() / (H - 1)))
        return X, Y

    # ------------------------------
    # One training step on click history
    # ------------------------------
    def train_on_clicks(self, flat_clicks):
        """
        flat_clicks: [x0,y0,x1,y1,...] from marker.ai_point_list
        """
        batch = self._build_batch(flat_clicks)
        if batch is None:
            return

        features, labels, latest_idx = batch

        logits = self.forward_batch(features)
        loss, probs = self._loss(logits, labels)

        self.losses.append(loss.detach())
        cost = torch.stack(self.losses).mean()
        print(f"loss {loss.item():.6f}  cost {cost.item():.6f}")

        # Use latest example in batch for UI prediction
        x, y = self._argmax_for_example(probs.detach(), latest_idx)
        self.game.ai = [x, y]
        self.game.ai_history.append([x, y])

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()



### ==============================
### Prediction Wrapper
### ==============================

def predictions(predictor, marker, game):
    clicks = marker.ai_point_list
    num_coords = len(clicks)

    # Don't run again if no new clicks since last frame
    if predictor.clicks >= num_coords:
        return

    predictor.clicks = num_coords

    # need at least window_size+1 clicks to have one example
    if (num_coords // 2) <= predictor.window_size:
        return

    predictor.train_on_clicks(clicks)


### ==============================
### Heatmap
### ==============================

def draw_heatmap(game):
    coords = game.ai
    size = (game.window.width, game.window.height)
    surface = pygame.Surface(size, pygame.SRCALPHA)
    surface.fill((255, 255, 255, 255))
    for i in range(4):
        game.draw.circle(surface, (255, 253, 253), coords, (4 * i))
        game.heatmap.blit(surface, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
    game.window.screen.blit(game.heatmap, (0, 0))


### ==============================
### Main
### ==============================

def main():
    frame = 0
    fps = 60
    pygame.init()
    marker = PointMarker()

    try:
        with open("clicks.json", "r") as f:
            marker.previous_clicks = json.load(f)
        print("Loaded previous clicks")
    except:
        print("No click file")

    prev = len(marker.previous_clicks)

    size = (900, 500)
    game = Game(
        ai=[450, 250],
        ai_history=[],
        heatmap=pygame.Surface(size, pygame.SRCALPHA),
        draw=pygame.draw,
        window=Window(size[0], size[1]),
        clock=pygame.time.Clock()
    )
    game.heatmap.fill((255, 255, 255))
    game.window.screen = pygame.display.set_mode(size, pygame.SRCALPHA)
    predictor = NNPredictor(game)
    background = pygame.Surface(size)
    background.fill((100, 150, 200))

    while True:
        game.window.screen.fill((255, 255, 255, 255))

        for event in pygame.event.get():
            marker.event_hook(event)
            if event.type == pygame.QUIT:
                with open("clicks.json", "w") as f:
                    json.dump(marker.previous_clicks, f)
                pygame.quit()
                sys.exit()

        # replay old clicks if present
        if frame % 7 == 0 and prev > marker.previous_clicks_position:
            click = marker.previous_clicks[marker.previous_clicks_position]
            marker.previous_clicks_position += 1
            marker.add((click[0], click[1]))

        game.window.screen.blit(background, (0, 0))
        draw_heatmap(game)

        game.draw.circle(
            game.window.screen,
            (240, 220, 120),
            (game.ai[0], game.ai[1]),
            20
        )

        marker.draw(game)
        predictions(predictor, marker, game)

        pygame.display.flip()
        game.clock.tick(fps)
        frame += 1


if __name__ == "__main__":
    main()
