# conda install -c conda-forge cdsapi

import cdsapi
import calendar

c = cdsapi.Client()

hours_in_day = [f'{x:02d}:00' for x in range(0, 24)]

def days_in_month(year: int, month: int) -> int:
    """Return the number of days in a given month of a given year."""
    num_days = calendar.monthrange(year, month)[1]
    return [str(day).zfill(2) for day in range(1, num_days + 1)]


def download_era5_land_runoff(client: cdsapi.Client, year: int, month: int) -> None:
    client.retrieve(
        'reanalysis-era5-land',
        {
            'format': 'netcdf.zip',
            'variable': 'runoff',
            'year': year,
            'month': str(month).zfill(2),
            'day': days_in_month(year, month),
            'time': hours_in_day
        },
        target=f'{year}_{str(month).zfill(2)}_era5land_hourly.netcdf.zip'
    )


def download_era5_runoff(client: cdsapi.Client, year: int, month: int) -> None:
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': 'runoff',
            'year': year,
            'month': str(month).zfill(2),
            'day': days_in_month(year, month),
            'time': hours_in_day
        },
        target=f'{year}_{str(month).zfill(2)}_era5_hourly.nc'
    )

if __name__ == '__main__':
    c = cdsapi.Client()

    # date logic to determine what dates should be attempted for download
    today = calendar.datetime.datetime.now()
    # era5 has 5+ days lag so subtract 5 days from today to get last expected era5 date.
    today = today - calendar.datetime.timedelta(days=5)
    # we want to download full months only so go to the previous month
    last_full_year = today.year if today.month > 1 else today.year - 1
    last_full_month = today.month - 1 if today.month > 1 else 12
    print(f'Last full year: {last_full_year}')
    print(f'Last full month: {last_full_month}')
    for year in range(1940, last_full_year + 1):
        for month in range(1, 13):
            if year == today.year and month > last_full_month:
                break
            download_era5_runoff(c, year, month)
