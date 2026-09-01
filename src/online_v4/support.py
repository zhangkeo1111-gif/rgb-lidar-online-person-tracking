from __future__ import division, absolute_import
import hashlib
import json
import math
import os
import site
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F
__all__ = ['osnet_x1_0']

class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, IN=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, groups=groups)
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False, gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError('Unknown gate activation: {}'.format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x

class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(self, in_channels, out_channels, IN=False, bottleneck_reduction=4, **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels))
        self.conv2c = nn.Sequential(LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels))
        self.conv2d = nn.Sequential(LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels))
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)

class OSNet(nn.Module):
    """Omni-Scale Network.
    
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. TPAMI, 2021.
    """

    def __init__(self, num_classes, blocks, layers, channels, feature_dim=512, loss='softmax', IN=False, **kwargs):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1], reduce_spatial_size=True, IN=IN)
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(self.feature_dim, channels[3], dropout_p=None)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(self, block, layer, in_channels, out_channels, reduce_spatial_size, IN=False):
        layers = []
        layers.append(block(in_channels, out_channels, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))
        if reduce_spatial_size:
            layers.append(nn.Sequential(Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return (y, v)
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))

def osnet_x1_0(num_classes=1000, pretrained=False, loss='softmax', **kwargs):
    if pretrained:
        raise ValueError('online_v3 only accepts its bundled frozen ReID weights')
    model = OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2], channels=[64, 256, 384, 512], loss=loss, **kwargs)
    return model
ROOT = Path(__file__).resolve().parents[2]
TEMPORAL_NPZ = ROOT / 'assets/priors/background/temporal_background_full_trainfit979.npz'
TEMPORAL_JSON = ROOT / 'assets/priors/background/temporal_background_full_trainfit979.json'
OCCUPANCY = ROOT / 'assets/priors/1_occupancy_xy_points.json'
REGISTRATION_CONFIG_PATH = ROOT / 'configs/projection_residual_audit.json'
TEMPORAL_SHA256 = '10223df84a250cb79d9bb79191a70ac54e390e39ac638c72951e1c4acb486e89'
REGISTRATION_CONFIG = json.loads(REGISTRATION_CONFIG_PATH.read_text(encoding='utf-8'))
PHYSICAL_PROJECTION = {
    'du_px': float(REGISTRATION_CONFIG['physical_baseline_du_px']),
    'dv_px': float(REGISTRATION_CONFIG['physical_baseline_dv_px']),
    'mode': 'RAW_PHYSICAL_CALIBRATION_BASELINE',
    'calibration_status': 'NOT_PROVEN_FINAL',
    'audit_candidate_writeback': False,
}
EMPIRICAL_INFERENCE_REGISTRATION = {
    'enabled': bool(REGISTRATION_CONFIG['empirical_inference_enabled']),
    'du_px': float(REGISTRATION_CONFIG['empirical_inference_du_px']),
    'dv_px': float(REGISTRATION_CONFIG['empirical_inference_dv_px']),
    'scope': str(REGISTRATION_CONFIG['empirical_inference_scope']),
    'mode': 'SCENE_SPECIFIC_EMPIRICAL_PIXEL_REGISTRATION',
    'physical_calibration_update': False,
    'provenance': str(REGISTRATION_CONFIG['empirical_inference_provenance']),
}
DISPLAY_OVERLAY_REGISTRATION = {
    'du_px': 60.0,
    'dv_px': 17.0,
    'mode': 'SCENE01_DISPLAY_ONLY_TUNING',
    'inference_effect': False,
    'scope': 'SCENE01_ONLY',
    'provenance': 'TRAIN_FIT online clear-person cylinder-to-RGB-box median residual audit',
}
LEGACY_DISPLAY_REGISTRATION = {
    'enabled': bool(REGISTRATION_CONFIG['legacy_display_enabled']),
    'du_px': float(REGISTRATION_CONFIG['legacy_display_du_px']),
    'dv_px': float(REGISTRATION_CONFIG['legacy_display_dv_px']),
    'mode': 'LEGACY_VISUALIZATION_ONLY',
    'inference_effect': False,
    'provenance': str(REGISTRATION_CONFIG_PATH),
}
MIN_COMPONENT_POINTS = 3
TRACK_COMPONENT_GATE_M = 1.2
MAX_ASSIGNMENT_COST = 1.15
K = np.asarray([[638.7348022460938, 0.0, 631.7426147460938], [0.0, 637.01904296875, 376.0455017089844], [0.0, 0.0, 1.0]], float)
D = np.asarray([-0.05481167882680893, 0.06478200852870941, -0.0008851818274706602, -0.00027690528077073395, -0.020605145022273064], float)
T_RSLIDAR_FROM_COLOR = np.asarray([[1.0, 0.0, 0.0, 0.04], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -0.07], [0.0, 0.0, 0.0, 1.0]], float)
T_COLOR_FROM_OPTICAL = np.asarray([[0.0007963267107333194, 0.0007963264582435126, 0.99999936586377, 0.0], [-0.9999996829318347, 6.341362301376385e-07, 0.0007963264582435126, 0.0], [5.551115123125783e-17, -0.9999996829318347, 0.0007963267107333194, 0.0], [0.0, 0.0, 0.0, 1.0]], float)
T_ANNOTATED_FROM_RSLIDAR = np.asarray([[0.965925826271113, 4.13984163700004e-11, 0.25881904516953275, -3.686162486360445e-11], [-3.9411981290630505e-11, 0.9999999999999998, -1.2864071446132213e-11, 1.1739287320011726e-10], [-0.2588190451695328, 2.2251567787446797e-12, 0.9659258262711125, 5.7131188668790855e-11], [0.0, 0.0, 0.0, 1.0]], float)
GROUND_NORMAL = np.asarray([0.004778112556410801, -0.01588542865538492, 0.9998624019318024], float)
GROUND_D = 2.1782561937946485

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def ground_z(x: float, y: float, normal: np.ndarray, plane_d: float) -> float:
    return float(-(normal[0] * x + normal[1] * y + plane_d) / normal[2])

def projection_assets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = np.linalg.inv(T_COLOR_FROM_OPTICAL) @ np.linalg.inv(T_RSLIDAR_FROM_COLOR) @ np.linalg.inv(T_ANNOTATED_FROM_RSLIDAR)
    return (transform, K.copy(), D.copy())

def project_points(points: np.ndarray, transform: np.ndarray, K: np.ndarray, D: np.ndarray,
                   du_px: float | None=None, dv_px: float | None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, float).reshape(-1, 3)
    camera = (transform @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    valid = np.all(np.isfinite(camera), axis=1) & (camera[:, 2] > 1e-08)
    pixels = np.full((len(points), 2), np.nan)
    if valid.any():
        pixels[valid] = cv2.projectPoints(camera[valid].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
        pixels[valid] += np.asarray([
            PHYSICAL_PROJECTION['du_px'] if du_px is None else float(du_px),
            PHYSICAL_PROJECTION['dv_px'] if dv_px is None else float(dv_px),
        ])
    return (pixels, valid, camera[:, 2])

def causal_support_delta_ms(rgb_timestamp_ns: int, lidar_timestamp_ns: int, sync_slop_ms: float) -> float:
    """Return past-RGB age; reject future or older-than-window support."""
    if rgb_timestamp_ns > lidar_timestamp_ns:
        raise ValueError('future RGB support is forbidden')
    delta_ms = (lidar_timestamp_ns - rgb_timestamp_ns) / 1000000.0
    if delta_ms > sync_slop_ms:
        raise ValueError('RGB support exceeds the synchronization window')
    return delta_ms

def load_frozen_background() -> tuple[set[tuple[int, int, int]], float, np.ndarray, dict]:
    receipt = json.loads(TEMPORAL_JSON.read_text(encoding='utf-8'))
    if receipt['source_split'] != 'TRAIN_FIT' or not receipt['frozen'] or receipt['validation_updates'] != 0 or (receipt['test_updates'] != 0):
        raise RuntimeError('Temporal background is not a frozen TRAIN_FIT-only asset')
    if sha256(TEMPORAL_NPZ) != TEMPORAL_SHA256:
        raise RuntimeError('Frozen temporal background hash changed')
    with np.load(TEMPORAL_NPZ) as value:
        keys = np.asarray(value['keys'], np.int32)
        counts = np.asarray(value['counts'], np.int64)
        voxel_size = float(np.asarray(value['voxel_size_m']).reshape(-1)[0])
        frame_count = int(np.asarray(value['frame_count']).reshape(-1)[0])
        threshold = float(np.asarray(value['frequency_threshold']).reshape(-1)[0])
    static = {tuple(map(int, key)) for key, count in zip(keys, counts, strict=True) if count / frame_count >= threshold}
    payload = json.loads(OCCUPANCY.read_text(encoding='utf-8'))
    occupancy = np.asarray([[float(row['x']), float(row['y'])] for row in payload], np.float64)
    return (static, voxel_size, occupancy, receipt)

def xyxy(detection: dict) -> np.ndarray:
    x, y, w, h = map(float, detection['bbox'])
    return np.asarray([x, y, x + w, y + h], float)

def box_cost(pixel: np.ndarray, box: np.ndarray) -> float:
    width, height = (max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0))
    center = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    center_distance = float(np.linalg.norm((pixel - center) / [width, height]))
    outside_x = max(box[0] - pixel[0], 0.0, pixel[0] - box[2]) / width
    outside_y = max(box[1] - pixel[1], 0.0, pixel[1] - box[3]) / height
    return 0.7 * math.hypot(outside_x, outside_y) + 0.3 * center_distance

def assign_tracks(tracks: list[dict], boxes: list[dict], pixels: np.ndarray, valid: np.ndarray) -> dict[int, int]:
    if not tracks or not boxes:
        return {}
    cost = np.full((len(tracks), len(boxes) + len(tracks)), 1.25, float)
    for i in range(len(tracks)):
        if valid[i]:
            for j, detection in enumerate(boxes):
                cost[i, j] = box_cost(pixels[i], xyxy(detection))
    rows, columns = linear_sum_assignment(cost)
    return {int(row): int(column) for row, column in zip(rows, columns) if column < len(boxes) and cost[row, column] <= MAX_ASSIGNMENT_COST}

def exclusive_point_owners(pixels: np.ndarray, boxes: list[np.ndarray], track_pixels: list[np.ndarray | None]) -> np.ndarray:
    owner = np.full(len(pixels), -1, int)
    best = np.full(len(pixels), np.inf, float)
    for index, box in enumerate(boxes):
        inside = (pixels[:, 0] >= box[0]) & (pixels[:, 0] <= box[2]) & (pixels[:, 1] >= box[1]) & (pixels[:, 1] <= box[3])
        if not inside.any():
            continue
        width, height = (max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0))
        target = track_pixels[index]
        if target is None or not np.isfinite(target).all():
            target = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        score = np.linalg.norm((pixels - target) / [width, height], axis=1)
        update = inside & (score < best)
        owner[update] = index
        best[update] = score[update]
    return owner

def adaptive_components(points: np.ndarray) -> list[np.ndarray]:
    if len(points) < MIN_COMPONENT_POINTS:
        return []
    tree = cKDTree(points)
    radii = 0.18 + 0.02 * np.linalg.norm(points[:, :2], axis=1)
    parent = np.arange(len(points))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index
    for index, neighbors in enumerate(tree.query_ball_point(points, radii)):
        for other in neighbors:
            if other > index:
                left, right = (find(index), find(other))
                if left != right:
                    parent[right] = left
    labels = np.asarray([find(index) for index in range(len(points))])
    return [points[labels == label] for label in np.unique(labels) if int(np.sum(labels == label)) >= MIN_COMPONENT_POINTS]

def cluster_center(points: np.ndarray) -> np.ndarray:
    lower = np.quantile(points, 0.05, axis=0)
    upper = np.quantile(points, 0.95, axis=0)
    return (lower + upper) / 2.0

def component_score(component: np.ndarray, box: np.ndarray, transform: np.ndarray, K: np.ndarray,
                    D: np.ndarray, du_px: float | None=None,
                    dv_px: float | None=None) -> tuple[float, dict]:
    center = cluster_center(component)
    pixel, valid, camera_depth = project_points(
        center[None], transform, K, D, du_px=du_px, dv_px=dv_px)
    if not valid[0] or camera_depth[0] <= 0:
        return (math.inf, {})
    width, height = (max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0))
    target = np.asarray([(box[0] + box[2]) / 2, box[1] + 0.54 * height])
    normalized_center = float(np.linalg.norm((pixel[0] - target) / [width, height]))
    span = np.ptp(component, axis=0)
    horizontal_span, vertical_span = (float(max(span[0], span[1])), float(span[2]))
    small_penalty = max(0.0, 0.1 - horizontal_span) + max(0.0, 0.1 - vertical_span)
    large_penalty = max(0.0, horizontal_span - 1.1) + 0.5 * max(0.0, vertical_span - 2.2)
    score = normalized_center + 0.35 * small_penalty + 0.45 * large_penalty - 0.055 * math.log1p(len(component))
    return (score, {'center_u': float(pixel[0, 0]), 'center_v': float(pixel[0, 1]), 'center_distance_norm': normalized_center, 'horizontal_span_m': horizontal_span, 'vertical_span_m': vertical_span, 'component_score': score})

def choose_component(components: list[np.ndarray], box: np.ndarray, transform: np.ndarray, K: np.ndarray,
                     D: np.ndarray, du_px: float | None=None,
                     dv_px: float | None=None) -> tuple[np.ndarray | None, dict]:
    ranked = []
    for component in components:
        score, details = component_score(
            component, box, transform, K, D, du_px=du_px, dv_px=dv_px)
        if math.isfinite(score):
            ranked.append((score, -len(component), component, details))
    if not ranked:
        return (None, {})
    ranked.sort(key=lambda item: (item[0], item[1]))
    _, _, component, details = ranked[0]
    return (component, {**details, 'candidate_components': len(ranked)})
ADAPTIVE_CC_CUDA = '\nextern "C" {\n__device__ __forceinline__ int root_of(const int* parent, int value) {\n    int next = parent[value];\n    while (next != value) { value = next; next = parent[value]; }\n    return value;\n}\n__global__ void hook_pairs(const double* points, const int* group, const int n,\n                           int* parent, int* changed) {\n    const long long pair = (long long)blockDim.x * blockIdx.x + threadIdx.x;\n    if (pair >= (long long)n*n) return;\n    const int i = pair / n, j = pair - (long long)i*n;\n    if (i >= j || group[i] != group[j]) return;\n    const double dx=points[3*i]-points[3*j];\n    const double dy=points[3*i+1]-points[3*j+1];\n    const double dz=points[3*i+2]-points[3*j+2];\n    const double radius=0.18 + 0.020*sqrt(points[3*i]*points[3*i] + points[3*i+1]*points[3*i+1]);\n    if (dx*dx + dy*dy + dz*dz > radius*radius) return;\n    const int ri=root_of(parent, i), rj=root_of(parent, j);\n    if (ri == rj) return;\n    const int high=ri > rj ? ri : rj, low=ri > rj ? rj : ri;\n    if (atomicMin(parent + high, low) > low) atomicExch(changed, 1);\n}\n__global__ void compress_roots(const int n, int* parent, int* changed) {\n    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n    if (i >= n) return;\n    const int root=root_of(parent, i);\n    if (parent[i] != root) { parent[i]=root; atomicExch(changed, 1); }\n}\n__device__ __forceinline__ unsigned long long cell_code(const int x, const int y) {\n    const unsigned int ux=(unsigned int)(x ^ (int)0x80000000u);\n    const unsigned int uy=(unsigned int)(y ^ (int)0x80000000u);\n    return ((unsigned long long)ux << 32) | (unsigned long long)uy;\n}\n__device__ __forceinline__ int lower_bound_code(const unsigned long long* values,\n                                                 const int n, const unsigned long long key) {\n    int left=0, right=n;\n    while (left < right) { const int mid=(left+right)/2;\n        if (values[mid] < key) left=mid+1; else right=mid; }\n    return left;\n}\n__global__ void occupancy_near(const double* query, const int n,\n                               const double* occupancy, const unsigned long long* codes,\n                               const int occupancy_n, unsigned char* near) {\n    const int i=blockDim.x*blockIdx.x+threadIdx.x;\n    if (i >= n) return;\n    const double x=query[2*i], y=query[2*i+1];\n    const int cx=(int)floor(x/0.07), cy=(int)floor(y/0.07);\n    near[i]=0;\n    for (int dx=-1; dx<=1 && !near[i]; ++dx) for (int dy=-1; dy<=1 && !near[i]; ++dy) {\n        const unsigned long long key=cell_code(cx+dx, cy+dy);\n        int position=lower_bound_code(codes, occupancy_n, key);\n        while (position < occupancy_n && codes[position] == key) {\n            const double ox=occupancy[2*position], oy=occupancy[2*position+1];\n            const double ddx=x-ox, ddy=y-oy;\n            if (ddx*ddx + ddy*ddy <= 0.07*0.07) { near[i]=1; break; }\n            ++position;\n        }\n    }\n}\n}\n'

class GpuFrustumBackend:

    def __init__(self, static: set[tuple[int, int, int]], voxel_size: float, occupancy_xy: np.ndarray):
        global cp
        if os.name == 'nt':
            roots = [Path(value) / 'nvidia' for value in site.getsitepackages()]
            bins = [path for root in roots for path in (root / 'cuda_nvrtc' / 'bin', root / 'cuda_runtime' / 'bin')]
            valid = [path for path in bins if path.exists()]
            if valid:
                os.environ['PATH'] = ';'.join(map(str, valid)) + ';' + os.environ.get('PATH', '')
                runtime_roots = [root / 'cuda_runtime' for root in roots if (root / 'cuda_runtime').exists()]
                if runtime_roots:
                    os.environ.setdefault('CUDA_PATH', str(runtime_roots[0]))
                for path in valid:
                    os.add_dll_directory(str(path))
        import cupy as cp
        keys = np.asarray(sorted(static), np.int64)
        self.key_min = keys.min(axis=0)
        self.key_max = keys.max(axis=0)
        spans = self.key_max - self.key_min + 1
        self.stride_y = int(spans[2])
        self.stride_x = int(spans[1] * spans[2])
        codes = (keys[:, 0] - self.key_min[0]) * self.stride_x + (keys[:, 1] - self.key_min[1]) * self.stride_y + (keys[:, 2] - self.key_min[2])
        self.static_codes = cp.asarray(np.sort(codes), dtype=cp.int64)
        self.key_min_gpu = cp.asarray(self.key_min)
        self.key_max_gpu = cp.asarray(self.key_max)
        self.voxel_size = float(voxel_size)
        occupancy = np.asarray(occupancy_xy, np.float64)
        cells = np.floor(occupancy / 0.07).astype(np.int64)
        ux = (cells[:, 0] + 2 ** 31).astype(np.uint64)
        uy = (cells[:, 1] + 2 ** 31).astype(np.uint64)
        occupancy_codes = ux << np.uint64(32) | uy
        order = np.argsort(occupancy_codes, kind='stable')
        self.occupancy = cp.asarray(occupancy[order])
        self.occupancy_codes = cp.asarray(occupancy_codes[order])
        module = cp.RawModule(code=ADAPTIVE_CC_CUDA, options=('--std=c++11',))
        self.hook_pairs = module.get_function('hook_pairs')
        self.compress_roots = module.get_function('compress_roots')
        self.occupancy_near = module.get_function('occupancy_near')

    def static_masks(self, points: np.ndarray, ground_height: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, dict]:
        # Preserve the optimized legacy production path exactly.  The explicit
        # all-point evidence method below is used by report-only redesign audits.
        value = cp.asarray(np.asarray(points, np.float64))
        height = value[:, 2] - cp.asarray(np.asarray(ground_height, np.float64))
        keys = cp.floor(value / self.voxel_size).astype(cp.int64)
        in_bounds = cp.all((keys >= self.key_min_gpu) & (keys <= self.key_max_gpu), axis=1)
        codes = (keys[:, 0] - self.key_min_gpu[0]) * self.stride_x + (keys[:, 1] - self.key_min_gpu[1]) * self.stride_y + (keys[:, 2] - self.key_min_gpu[2])
        positions = cp.searchsorted(self.static_codes, codes)
        positions = cp.minimum(positions, len(self.static_codes) - 1)
        temporal = ~(in_bounds & (self.static_codes[positions] == codes))
        occupancy_keep = cp.ones(len(value), dtype=cp.bool_)
        extreme = temporal & ((height <= 0.28) | (height >= 1.9))
        extreme_indices = cp.flatnonzero(extreme)
        if len(extreme_indices):
            query = cp.ascontiguousarray(value[extreme_indices, :2])
            near = cp.empty(len(query), cp.uint8)
            threads = 256
            blocks = (len(query) + threads - 1) // threads
            self.occupancy_near((blocks,), (threads,), (query, np.int32(len(query)), self.occupancy, self.occupancy_codes, np.int32(len(self.occupancy)), near))
            occupancy_keep[extreme_indices] = ~near.astype(cp.bool_)
        removed = {'temporal_removed': int(cp.sum(~temporal).get()),
                   'occupancy_removed': int(cp.sum(temporal & ~occupancy_keep).get())}
        return cp.asnumpy(temporal), cp.asnumpy(occupancy_keep), removed

    def static_evidence(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return CPU-visible temporal keep and 7 cm occupancy-near masks."""
        value = cp.asarray(np.asarray(points, np.float64))
        keys = cp.floor(value / self.voxel_size).astype(cp.int64)
        in_bounds = cp.all((keys >= self.key_min_gpu) & (keys <= self.key_max_gpu), axis=1)
        codes = (keys[:, 0] - self.key_min_gpu[0]) * self.stride_x + (keys[:, 1] - self.key_min_gpu[1]) * self.stride_y + (keys[:, 2] - self.key_min_gpu[2])
        positions = cp.searchsorted(self.static_codes, codes)
        positions = cp.minimum(positions, len(self.static_codes) - 1)
        temporal = ~(in_bounds & (self.static_codes[positions] == codes))
        near = cp.zeros(len(value), cp.uint8)
        if len(value):
            query = cp.ascontiguousarray(value[:, :2])
            threads = 256
            blocks = (len(query) + threads - 1) // threads
            self.occupancy_near((blocks,), (threads,), (query, np.int32(len(query)), self.occupancy, self.occupancy_codes, np.int32(len(self.occupancy)), near))
        return cp.asnumpy(temporal), cp.asnumpy(near.astype(cp.bool_))

    def static_keep(self, points: np.ndarray, ground_height: np.ndarray) -> tuple[np.ndarray, dict]:
        temporal, occupancy, removed = self.static_masks(points, ground_height)
        return (temporal & occupancy, removed)

    @staticmethod
    def owners(pixels: np.ndarray, boxes: list[np.ndarray]) -> np.ndarray:
        if not len(pixels) or not boxes:
            return np.full(len(pixels), -1, np.int32)
        p = cp.asarray(np.asarray(pixels, np.float64))
        b = cp.asarray(np.asarray(boxes, np.float64))
        inside = (p[:, None, 0] >= b[None, :, 0]) & (p[:, None, 0] <= b[None, :, 2]) & (p[:, None, 1] >= b[None, :, 1]) & (p[:, None, 1] <= b[None, :, 3])
        wh = cp.maximum(b[:, 2:4] - b[:, 0:2], 1.0)
        targets = (b[:, 0:2] + b[:, 2:4]) / 2.0
        score = cp.linalg.norm((p[:, None, :] - targets[None, :, :]) / wh[None, :, :], axis=2)
        score = cp.where(inside, score, cp.inf)
        owner = cp.argmin(score, axis=1).astype(cp.int32)
        owner[~cp.any(inside, axis=1)] = -1
        return cp.asnumpy(owner)

    def components(self, points: np.ndarray, owners: np.ndarray, box_count: int) -> list[list[np.ndarray]]:
        result: list[list[np.ndarray]] = [[] for _ in range(box_count)]
        selected = np.flatnonzero(owners >= 0)
        if not len(selected):
            return result
        p = cp.asarray(np.asarray(points[selected], np.float64))
        group = cp.asarray(np.asarray(owners[selected], np.int32))
        n = len(selected)
        threads = 256
        point_blocks = (n + threads - 1) // threads
        pair_blocks = (n * n + threads - 1) // threads
        parent = cp.arange(n, dtype=cp.int32)
        for _ in range(64):
            changed = cp.zeros(1, cp.int32)
            self.hook_pairs((pair_blocks,), (threads,), (p, group, np.int32(n), parent, changed))
            self.compress_roots((point_blocks,), (threads,), (np.int32(n), parent, changed))
            if int(changed.get()[0]) == 0:
                break
        else:
            raise RuntimeError('Adaptive CUDA connected components did not converge')
        labels = cp.asnumpy(parent)
        points_cpu = np.asarray(points[selected], np.float64)
        owners_cpu = owners[selected]
        for box_index in range(box_count):
            local = np.flatnonzero(owners_cpu == box_index)
            groups = []
            for label in np.unique(labels[local]):
                members = local[labels[local] == label]
                if len(members) >= MIN_COMPONENT_POINTS:
                    groups.append((int(selected[members].min()), points_cpu[members]))
            result[box_index] = [component for _, component in sorted(groups)]
        return result

    @staticmethod
    def synchronize() -> None:
        cp.cuda.Stream.null.synchronize()

