# NyaQueue

Go-брокер сообщений с ML-оптимизированной балансировкой и автоконфигурацией.
Хранение через `tidwall/wal`, оффсеты в `bbolt`, транспорт gRPC.

## Запуск

Локально через Docker (поднимает брокер, producer, consumer, broker-exporter,
Prometheus и Grafana):

```bash
make compose-up
```

Затем открыть Grafana: <http://localhost:3000> (anonymous viewer, дашборд
"Queue Quality" в папке NyaQueue). Prometheus: <http://localhost:9091>.

Остановить:

```bash
make compose-down
```

Локальные бинари (без Docker):

```bash
make build
./bin/broker          -config config.yaml
./bin/broker-exporter -config config.yaml
./bin/producer        -config config.yaml
./bin/consumer        -config config.yaml
```

## Эксперименты

Прогон бенчмарков для ВКР, результат в `experiments/results/`:

```bash
make run-experiment
```

Сравнение с Apache Kafka (поднимает Kafka в отдельном профиле):

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
