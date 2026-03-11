# Data Format

The framework expects intraday futures bars with these columns:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

Optional:

- `instrument`

## Timestamp handling

- If timestamps are naive, they are assumed to be in `data.source_timezone`.
- All timestamps are converted into `data.exchange_timezone`.
- Session filters are then applied in exchange time.

## Event calendar format

The optional event CSV should contain:

- `date`
- `event_name`

Example:

```csv
date,event_name
2024-01-10,CPI
2024-01-31,FOMC
2024-02-02,NFP
```
