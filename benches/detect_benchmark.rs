use criterion::{criterion_group, criterion_main, Criterion};
use mtcnn_rs::trt_mtcnn::*;

mod perf;

// Generate a flamegraph during criterion run
// enable probing of calls: echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
// then run the test for 30s: cargo bench --bench detect_benchmark -- --profile-time=30

pub fn criterion_benchmark(c: &mut Criterion) {
    let mt = Mtcnn::new("./test_resources").unwrap();

    let img = image::open("test_resources/DSC_0003.JPG").unwrap();

    c.bench_function("mtcnn detect face", |b| b.iter(|| mt.detect(&img, 40)));
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(perf::FlamegraphProfiler::new(100));
    targets = criterion_benchmark
}

criterion_main!(benches);
