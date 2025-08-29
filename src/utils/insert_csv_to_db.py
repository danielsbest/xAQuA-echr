import os
import sqlite3
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
DB_PATH = os.path.join(DATA_DIR, 'all_experiments.db')

def csv_to_sqlite(csv_path, conn, table_name):
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)

def main():
    csv_files = [f for f in os.listdir(DATA_DIR) if f.startswith('e_') and f.endswith('.csv')]
    conn = sqlite3.connect(DB_PATH)
    try:
        for csv_file in csv_files:
            table_name = os.path.splitext(csv_file)[0]
            csv_path = os.path.join(DATA_DIR, csv_file)
            print(f'Loading {csv_file} into table {table_name}')
            csv_to_sqlite(csv_path, conn, table_name)
    finally:
        conn.close()

if __name__ == '__main__':
    main()
