# 基于多种深度学习经典模型的图像识别系统

一个前后端分离的图像识别平台：**Vue 3 + Vite + Element Plus + ECharts** 做前端，
**Python + Django + DRF + PyTorch** 做后端，内置 **10 种经典卷积神经网络**，
支持在线训练、单图识别、多模型对比、批量识别、评估指标可视化与识别历史统计。

所有网络结构均为**从零实现**（不依赖任何在线预训练权重），因此在完全离线的环境下也能跑通
「数据集 → 训练 → 评估 → 推理 → 统计」的完整闭环；同时提供一键生成的合成演示数据集，
训练一个 LeNet-5 只需十几秒，训练一个 ResNet-18 到 90% 准确率也只需约 100 秒（纯 CPU）。

---

## 一、功能总览

| 模块 | 能力 |
| --- | --- |
| 系统概览 | 数据集 / 模型 / 识别次数 / 平均准确率统计，识别趋势折线图，结构准确率排行，最优模型与环境信息 |
| 用户登录 | 独立的登录页（新疆师范大学标识），Token 鉴权，路由守卫，登录态持久化，修改密码、安全退出；除登录与健康检查外全部接口需要登录 |
| 角色权限 | 两级角色：**管理员**可管理数据集、发起训练、管理模型、清空记录与维护账号；**普通用户**只能识别图片、查看模型库与识别历史。前端菜单与按钮按角色显示，后端接口同步强制校验 |
| 用户管理 | 管理员专用页面：新建用户、编辑资料、切换角色、重置密码、启用/停用、删除；含「不能删除自己 / 至少保留一个管理员」等安全约束 |
| 图像识别 | 拖拽上传或使用内置样例图，选择已训练模型，输出 Top-K 概率、置信度环、Softmax 全类别概率饼图、单张耗时 |
| 批量识别 | 一次上传最多 20 张图片，使用同一模型批量推理，结果表格展示缩略图、预测类别、置信度与耗时 |
| 模型对比 | 同一张图片同时送入多个模型，给出多数投票共识结果、一致率、各模型 Top-3 概率对比柱状图与耗时对比 |
| 模型训练 | 图形化配置超参数（结构 / 数据集 / 输入尺寸 / 轮数 / 批大小 / 学习率 / 优化器 / 策略 / 增强 / 设备），后台线程训练，实时进度条、Loss/Acc 双轴曲线、训练日志滚动、可中止可删除 |
| 模型库 | 10 种经典结构卡片（年份 / 流派 / 参数量 / CPU 推理耗时 / 特点），已训练模型管理（设为默认、重新评估、下载 .pt、删除），混淆矩阵热力图、分类别精确率/召回率/F1、训练曲线、参数量-准确率散点图 |
| 数据集 | 一键生成演示数据集、上传 zip 自定义数据集（ImageFolder 结构）、类别分布饼图、样本预览、统计刷新 |
| 识别历史 | 按模型 / 类别 / 来源 / 文件名筛选，分页浏览，删除与清空，每日识别量、类别分布、模型使用分布统计 |
| 操作日志 | 右下角「操作日志」浮窗实时记录页面访问与识别 / 训练 / 模型 / 数据集 / 用户操作，展示成功失败、耗时与关键结果，支持按模块筛选、搜索、导出与清空 |

---

## 二、内置的 10 种经典模型

| 网络 | 年份 | 流派 | 默认输入 | 关键思想 |
| --- | --- | --- | --- | --- |
| LeNet-5 | 1998 | CNN 奠基 | 32 | 局部感受野 + 权值共享，首个成功落地的卷积网络 |
| AlexNet | 2012 | 深度 CNN 起点 | 224 | ReLU、Dropout、数据增强、GPU 训练，ImageNet 夺冠 |
| VGG-11 | 2014 | 深度堆叠 | 224 | 全部使用 3×3 小卷积核堆叠 |
| VGG-16 | 2014 | 深度堆叠 | 224 | 13 卷积 + 3 全连接，迁移学习经典骨干 |
| GoogLeNet | 2014 | 多尺度并行 | 224 | Inception 模块、1×1 卷积降维、辅助分类器 |
| ResNet-18 | 2015 | 残差学习 | 224 | 残差连接解决深层网络退化 |
| ResNet-50 | 2015 | 残差学习 | 224 | Bottleneck 结构，工业界事实标准骨干 |
| DenseNet-121 | 2016 | 密集连接 | 224 | 层间密集连接，特征复用、缓解梯度消失 |
| MobileNetV2 | 2018 | 轻量化 | 224 | 倒残差 + 深度可分离卷积，移动端部署 |
| SqueezeNet | 2016 | 轻量化 | 224 | Fire 模块压缩参数量到 1.2M |

模型定义全部位于 [`backend/ml/zoo.py`](backend/ml/zoo.py)，元信息注册表位于
[`backend/ml/registry.py`](backend/ml/registry.py)。每个网络的末端都使用自适应池化，
因此同一个结构既支持 32×32 的小数据集，也支持 224×224 的 ImageNet 风格输入。

---

## 三、快速开始

### 0. 两种角色的权限一览

| 功能 | 普通用户 student | 管理员 admin |
| --- | --- | --- |
| 系统概览、图像识别、多模型对比 | ✅ | ✅ |
| 模型库（查看指标、下载权重） | ✅ | ✅ |
| 识别历史（查询、统计） | ✅ | ✅ |
| 模型训练（发起 / 中止 / 删除） | ❌ | ✅ |
| 数据集（上传 / 生成 / 删除 / 刷新） | ❌ | ✅ |
| 模型管理（激活 / 重新评估 / 删除） | ❌ | ✅ |
| 清空识别记录 / 删除记录 | ❌ | ✅ |
| 用户管理页面与接口 | ❌ | ✅ |
| Django 后台 `/admin/` | ❌ | ✅ |

普通用户访问受限页面会被前端路由守卫拦回系统概览；即使绕过界面直接调接口，后端也会返回
`403 {"detail": "该操作需要管理员权限..."}`，权限校验以后端为准。

### 1. 环境要求

- Python 3.10+（推荐 3.10 / 3.11）
- Node.js 18+
- 可选：支持 CUDA 的 GPU；本机为 Apple Silicon 时可尝试 `mps` 设备（默认 `auto` 自动探测）

### 2. 后端启动

```bash
# 在项目根目录
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd backend
python manage.py migrate                 # 初始化数据库
python manage.py create_default_user     # 创建登录账号（admin/admin123、student/123456）
python manage.py build_demo_dataset      # 生成演示数据集（6 类几何图形）
python manage.py runserver 8000          # 启动 API 服务
```

默认账号（可在 `backend/config/settings.py` 的 `DEMO_ACCOUNTS` 中修改，或执行
`python manage.py create_default_user --reset` 重置密码）：

| 账号 | 密码 | 角色 | 说明 |
| --- | --- | --- | --- |
| admin | admin123 | 管理员 | 数据集的增删、发起训练、模型管理、清空记录、用户管理，也可登录 Django Admin（/admin/） |
| student | 123456 | 普通用户 | 识别图片、多模型对比、浏览模型库与识别历史（无任何管理权限） |

建议答辩时分别用这两个账号登录演示，界面差异一目了然；新增账号可在「用户管理」页面完成。

登录页还会从 `/api/auth/config/` 读取站点信息，把 `SHOW_DEMO_ACCOUNT` 设为 `False`
（或设置环境变量 `IR_SHOW_DEMO_ACCOUNT=0`）即可隐藏页面上的演示账号提示。

登录页默认使用内置的占位校徽；若要换成学校官方校徽，把图片命名为 `logo.png` 放到
`frontend/public/` 目录后重新构建即可（`npm run build`）。

可选：命令行方式训练（等价于前端「模型训练」页面）

```bash
python manage.py train -a resnet18 -d 演示数据集-几何图形 -e 12 --image-size 64 --lr 0.01
python manage.py init_demo --train       # 一键：生成数据集 + 训练 lenet5/alexnet/resnet18
```

### 3. 前端启动

```bash
cd frontend
npm install
npm run dev        # 开发模式：http://localhost:5173（已配置 /api 与 /media 代理到 8000）
npm run build      # 生产构建：产物在 frontend/dist，Django 会自动托管，访问 http://127.0.0.1:8000 即可
```

> 生产模式下不需要单独启动前端服务：`npm run build` 后 Django 会接管 `frontend/dist`，
> 同时把前端路由回退到 `index.html`，实现单端口部署。

### 4. 第一次使用建议流程

1. 打开「数据集」页面 → 点击 **生成演示数据集**（约 2 秒，产出 1440 张 6 类几何图形）。
2. 打开「模型训练」页面 → 选择 `LeNet-5` → 点击 **开始训练**（CPU 下十几秒完成）。
3. 打开「图像识别」页面 → 点击内置样例图 → 选择刚训练好的模型 → **开始识别**。
4. 再训练一个 `ResNet-18`，回到「模型对比」页面就能看到两个模型的预测差异与共识结果。

### 5. 切换到 MySQL（可选）

系统默认使用 SQLite（零配置、便于演示）。如果要部署成多人使用的环境，可切换到 MySQL，
**业务代码无需改动**，只改环境变量。推荐用 Docker 运行 MySQL（不污染本机环境，删起来干净）。

#### 方式一：Docker（推荐）

```bash
# ① 启动 MySQL 8.0 容器并等待就绪（首次会拉取镜像，约 600MB）
docker compose up -d
docker compose ps                       # STATUS 显示 (healthy) 即可用

# ② 建表 + 迁移现有 SQLite 数据（导出 → 建表 → 导入 → 校验）
bash scripts/migrate_to_mysql.sh

# ③ 以 MySQL 模式启动后端
cd backend
IR_DB_ENGINE=mysql IR_DB_USER=ir_user IR_DB_PASSWORD=ir_pass_2026 \
python manage.py runserver 8000
```

数据库连接信息写在 `docker-compose.yml` 里（库名 `image_recognition`，账号 `ir_user / ir_pass_2026`，
root 密码 `root_pass_2026`），服务端字符集已固定为 `utf8mb4`，数据通过数据卷 `mysql_data` 持久化。

常用命令：

```bash
docker compose stop            # 停止（数据保留）
docker compose start           # 再次启动
docker compose logs -f mysql   # 查看日志
docker compose down            # 删除容器与网络（数据卷保留）
docker compose down -v         # 连数据一起删除（清空数据库，慎用）
```

也可以一条命令完成「启动 + 迁移」：`bash scripts/mysql_docker_up.sh --migrate`

#### 方式二：本机安装 MySQL

```bash
brew install mysql && brew services start mysql
mysql -u root < scripts/mysql_setup.sql     # 建库 + 建专用账号（utf8mb4）
bash scripts/migrate_to_mysql.sh
```

可配置的环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `IR_DB_ENGINE` | `sqlite` | 设为 `mysql` 即切换到 MySQL |
| `IR_DB_NAME` | `image_recognition` | 数据库名 |
| `IR_DB_USER` / `IR_DB_PASSWORD` | `ir_user` / `ir_pass_2026` | 账号密码（生产环境请修改） |
| `IR_DB_HOST` / `IR_DB_PORT` | `127.0.0.1` / `3306` | 连接地址 |

想切回 SQLite 时，去掉 `IR_DB_ENGINE` 环境变量即可，原来的 `backend/db.sqlite3` 仍然保留。
驱动方面项目已内置纯 Python 的 `pymysql`（免编译）；若安装过 MySQL 本机开发包，也可以
`pip install mysqlclient` 获得更好的性能，程序会优先使用它。

`scripts/migrate_to_mysql.sh` 会先把 SQLite 数据导出到 `work/sqlite_dump.json`，
再在 MySQL 上建表并导入，最后打印各表行数校验，整个过程可重复执行。

---

## 四、目录结构

```
recognition/
├── backend/                          # Django 后端
│   ├── config/                       # settings / urls / wsgi / asgi / SPA 托管
│   ├── apps/recognition/
│   │   ├── models.py                 # Dataset / TrainingRun / TrainedModel / PredictionRecord
│   │   ├── serializers.py            # 请求校验与响应结构
│   │   ├── views/                    # auth / architecture / dataset / training / model_registry / recognize / record / dashboard
│   │   ├── permissions.py            # 角色权限：IsAdminOrReadOnly / IsAdmin（管理员=is_staff）
│   │   ├── management/commands/      # build_demo_dataset / train / init_demo / create_default_user
│   │   └── migrations/
│   ├── ml/                           # 算法层（与 Django 解耦，可单独使用）
│   │   ├── zoo.py                    # 10 种经典网络的从零实现
│   │   ├── registry.py               # 模型元信息与参数量/耗时统计
│   │   ├── datasets.py               # ImageFolder 扫描、分层划分、数据增强、演示数据集生成
│   │   ├── engine.py                 # 训练/验证循环、优化器与学习率策略
│   │   ├── metrics.py                # 混淆矩阵、精确率/召回率/F1
│   │   ├── predictor.py              # 权重缓存、单图推理、多模型对比推理
│   │   └── trainer.py                # 后台训练线程管理与进度回写
│   ├── data/                         # datasets/（数据集）、weights/（权重）——运行时生成
│   ├── media/                        # 上传的识别图片
│   └── manage.py
├── frontend/                         # Vue 3 前端
│   ├── src/api/                      # axios 封装（自动附加 Token、401 自动跳登录）
│   ├── src/store/                    # Pinia 全局状态（app / auth）
│   ├── src/router/                   # 路由 + 登录守卫
│   ├── src/layouts/                  # 主框架布局（侧边栏 + 顶栏 + 用户菜单）
│   ├── src/components/               # StatCard / ProbBars / EChart
│   ├── src/views/                    # Login / Dashboard / Recognize / Compare / Training / Models / Datasets / Records / Users
│   ├── public/                       # 静态资源（放入 logo.png 可替换登录页校徽）
│   └── vite.config.js
├── docs/设计文档.md                   # 需求分析、总体设计、算法设计、测试与结论
├── docker-compose.yml                # Docker 版 MySQL 8.0（可选）
├── scripts/                          # setup / start-backend / start-frontend / mysql 相关脚本
└── requirements.txt
```

---

## 五、API 一览（前缀 `/api/`）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/auth/login/` | 登录，返回 Token 与用户信息（无需鉴权） |
| POST | `/api/auth/logout/` | 退出登录（注销当前 Token） |
| GET | `/api/auth/profile/` | 当前用户信息（含识别次数） |
| PATCH | `/api/auth/profile/` | 修改昵称与邮箱 |
| POST | `/api/auth/password/` | 修改密码，旧 Token 全部失效并签发新 Token |
| GET | `/api/auth/config/` | 登录页配置：校名、系统名、演示账号（无需鉴权） |
| GET | `/api/users/` | 用户列表（仅管理员，支持按用户名搜索、按角色筛选） |
| POST | `/api/users/` | 新建用户（仅管理员） |
| PATCH | `/api/users/{id}/` | 修改用户资料 / 角色 / 状态（仅管理员） |
| POST | `/api/users/{id}/reset-password/` | 重置密码，该账号所有登录状态失效（仅管理员） |
| POST | `/api/users/{id}/toggle-active/` | 启用 / 停用账号（仅管理员） |
| DELETE | `/api/users/{id}/` | 删除账号（仅管理员，自己 / 最后一个管理员不可删） |
| GET | `/api/health/` | 健康检查 |
| GET | `/api/system/` | 运行环境（Python / PyTorch / 设备 / 支持结构） |
| GET | `/api/dashboard/` | 首页聚合统计 |
| GET | `/api/architectures/` | 经典模型结构列表（含已训练情况） |
| GET | `/api/architectures/{key}/stat/` | 指定结构在给定类别数下的参数量、体积、CPU 推理耗时 |
| GET/POST | `/api/datasets/` | 数据集列表 / 上传 zip 数据集 |
| POST | `/api/datasets/build-demo/` | 一键生成演示数据集 |
| GET | `/api/datasets/{id}/samples/` | 样本图片地址列表 |
| GET/POST | `/api/training-runs/` | 训练任务列表 / 创建并启动训练 |
| GET | `/api/training-runs/{id}/progress/` | 轻量进度轮询（状态、进度、曲线） |
| GET | `/api/training-runs/{id}/log/` | 训练日志（`?tail=200`） |
| POST | `/api/training-runs/{id}/cancel/` | 中止训练 |
| GET | `/api/trained-models/` | 已训练模型列表 |
| POST | `/api/trained-models/{id}/activate/` | 设为该结构的默认推理模型 |
| POST | `/api/trained-models/{id}/evaluate/` | 用数据集重新评估，刷新混淆矩阵 |
| GET | `/api/trained-models/{id}/download/` | 下载权重 `.pt` |
| POST | `/api/recognize/` | 单图识别（`image` + `architecture`/`trained_model` + `top_k`） |
| POST | `/api/recognize/compare/` | 多模型对比（`image` + `architectures=a,b,c`） |
| POST | `/api/recognize/batch/` | 批量识别（`images` 多文件） |
| GET | `/api/recognize/sample/?label=star` | 生成一张合成样例图，便于无图快速体验 |
| GET | `/api/records/` | 识别历史（支持模型/类别/来源/时间筛选） |
| GET | `/api/records/stats/` | 识别统计（每日趋势、类别分布、模型分布、平均置信度与耗时） |

---

## 六、常见问题

**Q：登录页显示的是空校徽 / 想换成学校官方校徽？**
把官方校徽图片命名为 `logo.png` 放进 `frontend/public/`，重新执行 `npm run build` 即可自动替换；
不放置该文件时会使用内置的圆形占位徽标。校名、英文名、页脚等文案可在
`backend/config/settings.py` 的 `SITE_NAME` / `UNIVERSITY_NAME` / `UNIVERSITY_EN_NAME` 中修改。

**Q：忘记密码或需要新增账号？**
执行 `python manage.py create_default_user --reset` 重置默认账号密码；新增指定账号可执行
`python manage.py create_default_user --username 张三 --password 123456`
（加 `--superuser` 则同时获得 Django Admin 权限），也可以直接访问 `/admin/` 用管理员账号在后台维护用户。

**Q：没有 GPU 能跑吗？**
能。系统默认 `auto` 选择设备（CUDA → MPS → CPU）。演示数据集只有 64×64，CPU 上 LeNet-5
十几秒、ResNet-18 约 100 秒即可训练完成。VGG-16 参数约 138M，CPU 训练会明显变慢，建议
减小输入尺寸或降低轮数。

**Q：如何换成自己的数据集？**
把数据整理成 ImageFolder 结构：

```
my_dataset/
├── cat/*.jpg
├── dog/*.jpg
└── ...
```

打包成 zip 后在「数据集」页面上传即可；也支持 `my_dataset/train/<类别>` 与 `my_dataset/val/<类别>`
的显式划分方式（此时不会再次随机切分）。类目录名（英文）会自动作为类别标签。

**Q：为什么训练精度不高？**
检查三点：① 训练轮数是否足够（演示数据集建议 ≥10 轮）；② 输入尺寸是否过小（ResNet 系列建议 ≥64）；
③ 学习率是否合适（SGD 建议 0.01，Adam 建议 0.001）。也可以开启「数据增强」「标签平滑」，
或在样本不均衡时打开类别加权损失。

**Q：权重文件放在哪？**
`backend/data/weights/<数据集名>/<结构>_<数据集ID>_<时间戳>.pt`，文件内同时保存了
`state_dict`、类别列表、输入尺寸、训练历史与评估指标，可直接用 PyTorch 加载使用。

数据库里记录的是**相对 `backend/` 的相对路径**（如 `data/weights/演示数据集-几何图形/xxx.pt`），读取时才拼成绝对路径，
因此整个项目目录改名、搬迁或换机器都不需要改数据。若库里已经存过绝对路径，执行 `python manage.py make_paths_relative`
可一键批量转成相对路径（加 `--dry-run` 先预览，幂等，SQLite / MySQL 通用）。

**Q：`ml/` 目录可以脱离 Django 单独用吗？**
可以。`ml/` 只依赖 PyTorch / torchvision / PIL / numpy，不 import 任何 Django 模块，
适合直接拿来做课程实验或迁移到别的项目。

**Q：数据库用的是哪个？能换吗？**
默认 SQLite（`backend/db.sqlite3`，零配置），可以一键换成 MySQL。最省事的方式是用 Docker：
`docker compose up -d` 启动数据库，再执行 `bash scripts/migrate_to_mysql.sh` 迁移数据，
之后带上 `IR_DB_ENGINE=mysql` 等环境变量启动即可，详细步骤见上文「5. 切换到 MySQL（可选）」。
由于全部通过 Django ORM 访问、没有手写 SQL，换成 PostgreSQL 也只需改 `settings.py` 里的 `ENGINE`。
