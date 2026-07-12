# 原始排版还原（Layout Reconstruction）参考实现

> 适用场景：拿到 OCR 结构化结果（文字块 + bbox + 原图）后，在页面上按原版式"还原"出预览。
> 核心理念：**以结构化还原层为主** —— 默认在白底上用 bbox 绝对定位叠加出还原版面（文字/表格/公式/几何图），原图仅作为**可勾选的对照层**（勾选"显示原图"才叠到底层）。不重绘，是"还原 + 可选对照"。

---

## 一、统一数据契约（输入 / 输出）

整条链路围绕一个**统一的 JSON 结构**展开（三引擎产出都归一化到这里）：

```jsonc
OCRParseResponse {
  "ok": true,
  "engine": "paddleocr-vl",
  "pages": [
    {
      "page": 1,
      "width": 1000,        // 页面原始宽度(px)
      "height": 1400,       // 页面原始高度(px)
      "markdown": "整页 Markdown 文本（线性的，用于另一视图）",
      "elements": [         // 结构化元素列表（排版还原用这个）
        {
          "type": "title | subtitle | paragraph | formula | table | figure | header | footer",
          "bbox": [x1, y1, x2, y2],   // 像素，左上角原点，基于页面原始尺寸
          "content": "识别文本",
          "confidence": 0.98,          // 可选
          "latex": "E=mc^2",           // formula 类型才有
          "html": "<table>...</table>", // table 类型才有
          "caption": "图1 ..."          // figure 类型才有
        }
      ],
      "crops": [            // 裁剪图（base64 内联，用于 figure 还原）
        {
          "filename": "doc_page1_type3.png",
          "base64": "data:image/png;base64,iVBORw0KGgo...",
          "element_index": 3,            // ★ 关键：对应 elements[] 的下标
          "bbox": [x1, y1, x2, y2]       // 与对应元素 bbox 一致
        }
      ]
    }
  ]
}
```

**坐标系统约定（务必前后端一致）**
- `bbox = [x1, y1, x2, y2]`，单位**像素**，原点**左上角**。
- 基于**页面原始尺寸**（即 `width × height` 字段），不是显示尺寸。
- 前端换算百分比：`left = x1/width*100%`，其余同理 —— 这样缩放/响应式都不会错位。

### 1.1 坐标归一化（引擎差异，关键坑）

并非所有引擎都直接输出**像素坐标**。例如 **Unlimited-OCR**（baidu/Unlimited-OCR，基于 Qwen2-VL grounding）返回的是**归一化坐标（0~1000 空间）**，与输入图片的真实像素尺寸（如 691×922）完全不匹配。

**典型症状**：crop 裁剪图错位、套进旁边文字、甚至整张空白。

**后端自动校正**：`unlimited_ocr_client.py` 中 `_detect_and_scale_coords(elements, width, height)` 在生成 crops 前做检测与缩放：
- **判据**：任一 bbox 坐标 `> 图片宽/高` 或 `< 0` → 判定为归一化坐标系。
- **缩放**：`x *= width/1000`、`y *= height/1000`，把全部元素 bbox 转成真实像素坐标。
- 这样后续 crop 裁剪、前端叠加都基于一致像素系，无需前端做任何换算。

> ⚠️ 这个缩放必须发生在**后端生成 crops 之前**，否则按越界坐标裁出来的图就是错的（且该错误无法在前端补救——前端拿到的已经是错坐标）。

---

## 二、后端：从原图生成 crops（输入侧处理）

每个结构化元素除了 `bbox`，还要额外生成一张**从原图裁出来的小图**，供 `figure` 类（几何图、插图等）还原。

```python
from PIL import Image
import base64, io

def _decode_dims(image_data: str):
    """把 data URL 解码成 PIL 图，返回 (pil, w, h)"""
    _, data = (image_data.split(",", 1) + [""])[:2] if "," in image_data else ("", image_data)
    pil = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    return pil, pil.size[0], pil.size[1]

def _crop_to_base64(pil, bbox, pad: int = 3):
    """按 bbox 从原图裁一块 PNG，base64 内联返回；pad 防止贴边裁断"""
    w, h = pil.size
    x1 = max(0, int(bbox[0]) - pad)
    y1 = max(0, int(bbox[1]) - pad)
    x2 = min(w, int(bbox[2]) + pad)
    y2 = min(h, int(bbox[3]) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = pil.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

# 在 element 循环里：
elements = []
crops = []
for b_idx, blk in enumerate(blocks):
    bbox = [float(v) for v in blk["bbox"]]
    el = {"type": blk["type"], "bbox": bbox, "content": blk.get("content", ""), ...}
    elements.append(el)
    if pil is not None:
        crop_b64 = _crop_to_base64(pil, bbox)
        if crop_b64:
            crops.append({
                "filename": f"{doc_id}_page{page}_type{b_idx}.png",
                "base64": crop_b64,
                "element_index": len(elements) - 1,   # ★ 指向 elements[] 的下标
                "bbox": bbox,                          # 与元素 bbox 一致，兜底匹配用
            })
```

> **★ 关键经验**：`element_index` 和 `bbox` 必须跟着 crop 一起产出，否则前端没法把"图"对回"位置"。
> **★ 致命坑（我们踩过）**：如果后端用 FastAPI `response_model` 做响应序列化，而 schema 里**没声明** `element_index`/`bbox`，Pydantic 会按 schema 重新序列化并**悄悄丢弃**这两个字段 —— 前端拿到的 crop 只剩 `{filename, base64}`，怎么匹配都对不上。务必在 schema 里把这两个字段声明为 `Optional`。

---

## 三、前端：还原层渲染（核心，原图可勾选对照）

### 3.1 DOM 结构

```
.layout-wrap
 ├─ .layout-toolbar   （缩放滑块 / 重置 / 原图状态 / 两个勾选框）
 └─ .layout-view      （滚动容器，position:relative，浅色背景）
     └─ .layout-page  （position:relative; inline-block；默认白底=还原层；原图隐藏时仍由原图(visibility:hidden)或 aspect-ratio 撑开尺寸）
         ├─ <img class="bg">      ← 原图底图（默认隐藏，勾选"显示原图"才叠加，用于对照）
         ├─ .layout-box (文字)     ← 绝对定位，百分比坐标，默认带边框
         ├─ .layout-box (表格)
         └─ .layout-box (figure)   ← 内含 <img> 裁剪图
```

### 3.2 CSS（关键几条）

```css
.layout-view  { position:relative; overflow:auto; background:#f5f6f8; }   /* 滚动容器，浅灰底 */
.layout-page  { position:relative; display:inline-block; background:#fff; } /* 白底=还原层；不写死宽高 */
.layout-page > img.bg { display:block; min-width:200px; }                 /* 原图底图 */
/* ★ 默认隐藏原图：用 visibility 而非 display:none，既藏住原图又保住页面尺寸基准（bbox 百分比定位不会错位）；
   勾选"显示原图"后 .layout-page 加 .show-original 类才显示原图做对照 */
.layout-page:not(.show-original) > img.bg { visibility:hidden; }
.layout-box  { position:absolute; overflow:hidden;
               font-size:11px; line-height:1.2; color:#111; background:transparent;
               padding:1px 2px; box-sizing:border-box;
               /* 双色描边：白+黑 text-shadow，保证在白底还原层或原图底上都可读 */
               text-shadow:0 0 2px #fff,0 0 2px #fff,0 0 4px #fff,0 0 4px #000; }
.layout-box > img { display:block; width:100%; height:100%; object-fit:contain; border-radius:2px; }
.layout-box.show-border { outline:1px solid rgba(58,122,254,.95); background:rgba(58,122,254,.18); } /* 默认开启，勾勒版面结构 */
```

### 3.3 渲染算法（伪代码 → 可直接抄）

```javascript
function buildLayoutView(p, imgUrl) {
  const W = p.width || 1, H = p.height || 1;

  // 1) 原图底图（默认隐藏，勾选"显示原图"才叠加对照）
  const page = document.createElement('div'); page.className = 'layout-page';
  if (imgUrl) {
    const img = document.createElement('img'); img.className = 'bg'; img.src = imgUrl;
    img.onerror = () => status.textContent = '✗ 原图加载失败';
    page.appendChild(img);
  } else {
    // 无原图：按页面宽高比撑开还原层，保证 bbox 叠加仍有尺寸基准
    page.style.aspectRatio = W + ' / ' + H;
    page.style.width = '100%';
  }

  // 2) 预建 crop 索引（供 figure 匹配）
  const cropByIndex = {};                       // element_index → base64
  const cropByBbox = [];                        // [{bbox, base64}] 兜底
  for (const c of (p.crops || [])) {
    if (!c.base64) continue;
    if (typeof c.element_index === 'number') cropByIndex[c.element_index] = c.base64;
    if (Array.isArray(c.bbox) && c.bbox.length === 4) cropByBbox.push({ bbox: c.bbox, base64: c.base64 });
  }
  // bbox IoU 模糊匹配（引擎没给 element_index 时的兜底）
  function findCropBbox(x1, y1, x2, y2) {
    let best = null, bestIoU = 0;
    for (const cb of cropByBbox) {
      const ix1 = Math.max(x1, cb.bbox[0]), iy1 = Math.max(y1, cb.bbox[1]);
      const ix2 = Math.min(x2, cb.bbox[2]), iy2 = Math.min(y2, cb.bbox[3]);
      if (ix2 <= ix1 || iy2 <= iy1) continue;
      const inter = (ix2 - ix1) * (iy2 - iy1);
      const area = (x2 - x1) * (y2 - y1), cArea = (cb.bbox[2] - cb.bbox[0]) * (cb.bbox[3] - cb.bbox[1]);
      const iou = inter / (Math.min(area, cArea) || 1);
      if (iou > bestIoU) { bestIoU = iou; best = cb.base64; }
    }
    return best;
  }

  // 3) 遍历 elements，绝对定位叠加
  const els = p.elements || [];
  for (let ei = 0; ei < els.length; ei++) {
    const e = els[ei], b = e.bbox;
    if (!Array.isArray(b) || b.length !== 4) continue;
    const [x1, y1, x2, y2] = b;
    if (x2 <= x1 || y2 <= y1) continue;        // 非法框跳过

    const box = document.createElement('div'); box.className = 'layout-box show-border';  // 默认带边框，凸显还原结构
    box.style.left   = (x1 / W * 100) + '%';
    box.style.top    = (y1 / H * 100) + '%';
    box.style.width  = ((x2 - x1) / W * 100) + '%';
    box.style.height = ((y2 - y1) / H * 100) + '%';

    if (e.type === 'table' && e.html) {
      box.innerHTML = `<div class="lt">[表格]</div>${e.html}`;
    } else if (e.type === 'figure') {
      // ★ 优先 element_index 精确匹配，其次 bbox IoU 模糊匹配
      const imgSrc = cropByIndex[ei] || findCropBbox(x1, y1, x2, y2);
      if (imgSrc) {
        const img = document.createElement('img'); img.src = imgSrc;
        img.onerror = () => { img.style.display = 'none'; box.textContent = '🖼 ' + (e.caption || '加载失败'); };
        box.appendChild(img);
      } else {
        box.textContent = '🖼 ' + (e.caption || '');   // 无图降级为文字
      }
    } else {
      const body = (e.type === 'formula') ? ('⨎ ' + (e.latex || e.content || '')) : (e.content || '');
      box.textContent = body;
    }
    page.appendChild(box);
  }
  return page;
}
```

### 3.4 缩放（CSS transform，不影响布局计算）

```javascript
function setZoom(pct) {
  page.style.transform = 'scale(' + pct / 100 + ')';
  page.style.transformOrigin = 'top left';   // 从左上角缩放，避免偏移
}
// 滑块 oninput / 滚轮 onwheel / 重置按钮 都调 setZoom
```

> 缩放用 `transform: scale` 而不是改 width，因为百分比定位已经绑定在 `.layout-page` 上，transform 不改变子元素坐标基准，整套布局零计算成本。

### 3.5 「裁剪图」诊断 Tab（验证 crop 是否准）

测试台在每页 `Markdown` / `排版还原` 旁提供第三 tab **「裁剪图」**，把 `crops[]` 中**每一张识别后裁出的原图**直接网格陈列，供人工核对"裁图到底对不对"：

- 每张图标注 `#序号`、`type`（来自 `elements[].type`）、裁图 `WxH px`（由 bbox 换算）。
- 点击任意裁图可在新标签页打开原图大图；图片懒加载，加载失败自动淡出。
- 这是验证 §1.1 坐标问题最快的手段：**缩放正确的话，几何图形应被完整裁出**；若裁出空白/错位/套进相邻文字，多半是坐标没归一化（见 §1.1）或引擎本身未返回该区域。

---

## 四、踩坑清单（直接避坑）

| 现象 | 根因 | 解法 |
|---|---|---|
| figure 区域显示成**小方块/空白** | 后端 `response_model` 把 `element_index`/`bbox` 字段剥掉，前端匹配不到 crop | schema 里声明这两个 `Optional` 字段 |
| 高图被压成**细长条** | 给容器写死 `max-height`+`overflow:hidden`，把原图压扁了 | 去掉高度限制；`.layout-page` 用 `inline-block` 让 `<img>` 自然撑开；给 `bg` 设 `min-width:200px` 防过缩 |
| 原图底图**默认不显示** / 切换时 bbox 错位 | 用 `display:none` 把原图移出布局，页面尺寸塌为 0 | 用 `visibility:hidden` 隐藏（保住尺寸基准）；无原图时按 `aspect-ratio=W/H` + `width:100%` 撑开还原层 |
| 文字在底图上**看不清** | 纯色文字无描边 | `text-shadow` 双色描边（白+黑），白底还原层与原图底图都可读 |
| 引擎没返回 `element_index`，图对不上 | 兜底缺失 | 用 bbox **IoU** 模糊匹配 crop（见 `findCropBbox`） |
| 响应报文预览里**大图 base64 卡死浏览器** | 巨长单行字符串导致折行崩溃 | 预览时把 `crops[].base64` 截断显示 |
| **crop 错位 / 套进旁边文字 / 整张空白**（Unlimited-OCR 明显） | 引擎返回**归一化坐标（0~1000）**，与真实像素尺寸不符，按越界坐标裁原图被截断到边缘 | 后端 `_detect_and_scale_coords()` 自动检测并缩放到像素座标（见 §1.1），缩放必须在生成 crops 前完成 |
| 公式显示为 `\(...\)` 原始 LaTeX + 控制字符（Unlimited-OCR） | 引擎直接吐出原始 `\(` `\)` 包裹和控制字符（如 `\t`） | 后端 `_clean_latex()` 清洗：去 `\(\)`/`\[\]` 包裹 → `$...$`，并 strip 控制字符 |
| 同一段文字**叠加两层**（Unlimited-OCR 偶发） | 引擎把同一区域输出了两次（bbox 差 1px、内容差 1 字符） | 后端 `_dedup_elements()` 按"类型 + 量化 bbox + 去标点签名"去重近重复块 |

---

## 五、一句话总览

```
原图(base64) ──┐
               ├─→ 后端按 bbox 裁 crops[]（带 element_index）
结构化 elements[] ─┘
                       ↓ 统一 JSON
前端：白底还原层（elements[] 绝对定位百分比叠加，默认显示）
      ＋ 原图底图 <img.bg>（默认隐藏，勾选"显示原图"才叠加对照）
      ＋ figure 用 crop 内嵌 <img>
                       ↓
             排版还原（缩放 = CSS scale，零重排）
```

这套方案不依赖任何 OCR 引擎私有格式，只要上游能产出 `{width,height,elements:[{type,bbox,content,...}],crops:[{element_index,base64}]}` 就能直接套用。
