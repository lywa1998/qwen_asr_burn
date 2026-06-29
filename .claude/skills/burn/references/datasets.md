# Datasets and DataLoaders

Burn's data pipeline is two pieces:

- **`Dataset<I>`** — random-access collection of items.
- **`Batcher<I, O>`** — pure function that turns a `Vec<I>` into a batched tensor struct `O`.

The `DataLoaderBuilder` glues them together with multi-worker iteration.

## The `Dataset` trait

```rust
pub trait Dataset<I>: Send + Sync {
    fn get(&self, index: usize) -> Option<I>;
    fn len(&self) -> usize;
}
```

Fixed length, constant-time random access. No streaming, but you can wrap it with lazy transforms.

## Built-in datasets

In `burn::data::dataset::*`:

- `vision::MnistDataset`, `vision::CifarDataset`, `vision::ImageFolderDataset`
- `audio::SpeechCommandsDataset` (with `audio` feature)
- `SqliteDataset<I>` — fast random access from a SQLite file. The Hugging Face loader writes to this.
- `HuggingfaceDatasetLoader` — download + cache datasets from HuggingFace into SQLite (needs `sqlite` feature)
- `InMemDataset<I>` — wraps a `Vec<I>` for tests / small datasets

```rust
use burn::data::dataset::vision::MnistDataset;
let train = MnistDataset::train();   // downloads on first call
let test = MnistDataset::test();
println!("{}", train.len());         // 60000
let item = train.get(0).unwrap();    // MnistItem { image, label }
```

## Implementing your own `Dataset`

```rust
use burn::data::dataset::Dataset;

pub struct CsvDataset {
    rows: Vec<MyRow>,
}

impl Dataset<MyRow> for CsvDataset {
    fn get(&self, index: usize) -> Option<MyRow> {
        self.rows.get(index).cloned()
    }
    fn len(&self) -> usize { self.rows.len() }
}
```

For datasets too big to fit in RAM, back it with `SqliteDataset` or memory-mapped files — keep `get` O(1) and side-effect-free.

## Dataset transformations (lazy)

In `burn::data::dataset::transform::*`:

| Transform | Use it for |
| --------- | ---------- |
| `MapperDataset` | Apply a function to each item (normalization, augmentation, decoding bytes). |
| `SamplerDataset` | Sample with/without replacement; useful to fix an epoch size independent of the underlying length. |
| `SelectionDataset` | Take a specific list of indices; supports shuffling with a seed. |
| `ShuffledDataset` | Shuffle. Thin wrapper around `SelectionDataset`. |
| `PartialDataset` | Index range view. Use for train/val/test splits. |
| `ComposedDataset` | Concatenate multiple datasets into one. |
| `WindowsDataset` | Overlapping windows (time-series, audio). |

All lazy: no copies, no upfront iteration.

```rust
use burn::data::dataset::transform::{PartialDataset, ShuffledDataset};

let full = MyCsvDataset::load("data.csv");
let shuffled = ShuffledDataset::new(full, 42);
let len = shuffled.len();
let train = PartialDataset::new(shuffled.clone(), 0, len * 8 / 10);
let valid = PartialDataset::new(shuffled, len * 8 / 10, len);
```

A `MapperDataset` takes a `Mapper` trait impl:

```rust
use burn::data::dataset::transform::{Mapper, MapperDataset};

struct Normalize;
impl Mapper<RawItem, NormalizedItem> for Normalize {
    fn map(&self, item: &RawItem) -> NormalizedItem {
        NormalizedItem {
            data: (item.data.clone() - MEAN) / STD,
            label: item.label,
        }
    }
}

let normalized = MapperDataset::new(raw_dataset, Normalize);
```

## Batcher trait

```rust
pub trait Batcher<I, O>: Send + Sync + Clone {
    fn batch(&self, items: Vec<I>, device: &Device) -> O;
}
```

A batcher is **stateless and cheap to clone**. The dataloader clones it per worker. Don't put a model or any allocated tensors in the batcher — keep it pure.

```rust
use burn::{data::dataloader::batcher::Batcher, prelude::*};

#[derive(Clone, Default)]
pub struct MnistBatcher {}

#[derive(Clone, Debug)]
pub struct MnistBatch {
    pub images: Tensor<3>,
    pub targets: Tensor<1, Int>,
}

impl Batcher<MnistItem, MnistBatch> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &Device) -> MnistBatch {
        let images = items.iter()
            .map(|item| TensorData::from(item.image))
            .map(|data| Tensor::<2>::from_data(data, device))
            .map(|t| t.reshape([1, 28, 28]))
            .map(|t| ((t / 255.0) - 0.1307) / 0.3081)
            .collect();

        let targets = items.iter()
            .map(|item| Tensor::<1, Int>::from_data([item.label as i64], device))
            .collect();

        MnistBatch {
            images: Tensor::cat(images, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}
```

Two patterns to notice:

1. **Build a `Vec<Tensor<..>>` per item, then `Tensor::cat` once.** This is more efficient than appending to a running tensor.
2. **Normalize inside the batcher** rather than in the model's forward. Keeps the model independent of dataset statistics.

## DataLoader

```rust
use burn::data::dataloader::DataLoaderBuilder;

let dataloader = DataLoaderBuilder::new(MnistBatcher::default())
    .batch_size(64)
    .shuffle(seed)               // shuffle indices each epoch
    .num_workers(4)              // parallel workers; each clones the batcher
    .build(MnistDataset::train());

for batch in dataloader.iter() {
    let MnistBatch { images, targets } = batch;
    // ... train ...
}
```

`num_workers > 1` spawns a thread pool. Each worker pulls indices, fetches items via `dataset.get`, then calls `batcher.batch`. The batcher's `device` argument is the **augmentation device** — for best throughput, do augmentation on CPU and let the training thread copy onto the GPU device.

A persistent dataloader keeps workers alive across epochs (recent change — see `git log --oneline | grep dataloader`). That's the default with `DataLoaderBuilder`.

## Where to do augmentation

Three valid places:

1. **In the `Dataset` (via `MapperDataset`)** — runs once per `get` call, can be cached.
2. **In the `Batcher`** — runs after items are collected, before tensor concat. Use this for cropping, flipping, normalization.
3. **In the model's `forward`** — runs on the training device. Use this for normalization that requires running statistics (already done by `BatchNorm`).

If augmentation is heavy (random crops, color jitter), do it in the dataset or batcher on CPU and let the GPU focus on the model.

## HuggingFace datasets

With the `sqlite` feature:

```rust
use burn::data::dataset::source::huggingface::HuggingfaceDatasetLoader;

let dataset: SqliteDataset<DbPediaItem> = HuggingfaceDatasetLoader::new("fancyzhx/dbpedia_14")
    .dataset("train")
    .unwrap();
```

Downloads the dataset, converts it to a local SQLite file, and gives you a typed `Dataset`. The `Item` type just needs `serde::Deserialize`.

## Custom dataset example: CSV

```rust
use burn::data::dataset::Dataset;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct Row {
    pub features: Vec<f32>,
    pub label: usize,
}

pub struct CsvDataset {
    rows: Vec<Row>,
}

impl CsvDataset {
    pub fn load(path: &str) -> Self {
        let rows: Vec<Row> = csv::Reader::from_path(path)
            .unwrap()
            .deserialize()
            .filter_map(Result::ok)
            .collect();
        Self { rows }
    }
}

impl Dataset<Row> for CsvDataset {
    fn get(&self, index: usize) -> Option<Row> {
        self.rows.get(index).cloned()
    }
    fn len(&self) -> usize { self.rows.len() }
}
```

`Row` should be `Clone` for `get` to return owned values cheaply (or wrap heavy fields in `Arc`).

## Tips

- The dataset should be **deterministic and side-effect-free**. All randomness goes through `shuffle(seed)` on the dataloader, or explicit seeded sampling.
- The batcher must be **`Send + Sync + Clone`**. Anything inside it must be cheap to clone or the cloning shows up in your top frames.
- For variable-length items (text), pad inside the batcher or use `Tensor::cat` with padding masks. The model handles masking via attention modules.
- Don't compute statistics (mean/std) on the fly inside `batch` — precompute and bake them in as constants, or maintain them as `BatchNorm` running stats.
