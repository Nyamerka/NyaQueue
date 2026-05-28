# NyaQueue

Адаптивный брокер сообщений на Go с интеграцией методов машинного обучения.

## Архитектура

NyaQueue объединяет классические компоненты брокера (WAL, партиционирование, gRPC/HTTP-транспорт) с ML-модулями, работающими в замкнутом контуре обратной связи:

- **DQN-балансировщик** (QR-Dueling DQN) — выбирает партицию для каждого сообщения на основе текущей и прогнозной загрузки
- **DQN-планировщик** (Dueling DQN) — адаптивный порог приоритета доставки (τ ∈ {0…9})
- **TD3-оптимизатор** — фоновая подстройка 22 параметров конфигурации (хранение, сеть, батчирование, потребление) каждые 250 мс без остановки брокера
- **Конвейер предобработки** — LHS → PCA → Lasso для отбора значимых параметров
- **Backpressure** — авторегрессия AR(p) предсказывает загрузку, конечный автомат Open/Warn/Closed ограничивает приём превентивно

Хранение: `tidwall/wal` (batch-запись, fsync), оффсеты в `BadgerDB`, индекс приоритетов `PriorityIndex`.
Транспорт: gRPC (7 RPC) + HTTP/2 REST.

## Результаты

Экспериментальный стенд: 6 сценариев нагрузки × 6 алгоритмических конфигураций × 3 режима (in-process, HTTP, gRPC) + сравнение с Apache Kafka 3.6.0.

| Метрика | Значение |
|---------|----------|
| TD3 прирост λ | +2.5–3.6% (rate-limited) |
| TD3 overhead | <0.001% |
| p₅₀ vs Kafka (HTTP) | 1.4× ниже |
| p₉₅ vs Kafka (bursty) | 2.7× ниже |
| σᵨ все алгоритмы | <10⁻³ |

TD3 (фоновая оптимизация) эффективнее DQN (per-request инференс): DQN создаёт +5% к p₅₀, при K=32 деградирует на 21% по λ.

## Запуск

Docker (поднимает брокер, producer, consumer, broker-exporter, Prometheus и Grafana):

```bash
make compose-up
```

Grafana: <http://localhost:3000> (дашборд "Queue Quality"). Prometheus: <http://localhost:9091>.

```bash
make compose-down
```

Локальные бинари:

```bash
make build
./bin/broker          -config config.yaml
./bin/broker-exporter -config config.yaml
./bin/producer        -config config.yaml
./bin/consumer        -config config.yaml
```

## Эксперименты

Прогон бенчмарков, результат в `experiments/results/`:

```bash
make run-experiment
```

Сравнение с Apache Kafka:

```bash
make compose-kafka
make compose-experiment
```

## Разработка

```bash
make test
make bench
make lint
make proto
```

## Лицензия

MIT
