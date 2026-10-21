"""Загрузка и очистка данных Белстата «Экспорт/Импорт важнейших видов продукции».

Исходные файлы: широкий формат (строки - товары, столбцы - годы 2000-2021),
единица измерения «зашита» в название товара после последней запятой.
Числа хранятся как текст с неразрывным пробелом и десятичной запятой
(например '1\xa0196,8'), поэтому нужна явная нормализация.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

FILES = {
    "export": RAW / "export_tovarov_vazh-2021-1.xlsx",
    "import": RAW / "import_tovarov_vazh-2021-1.xlsx",
}

# Короткие названия для подписей на графиках
SHORT_NAMES = {
    "Тара пластмассовая": "Пластмассовая тара",
    "Лесоматериалы продольно-распиленные": "Пиломатериалы",
    "Плиты древесно-стружечные": "ДСП",
    "Химические волокна и нити": "Хим. волокна и нити",
    "Черные металлы": "Чёрные металлы",
    "Холодильники, морозильники и холодильное оборудование": "Холодильники",
    "Провода изолированные, кабели": "Провода и кабели",
    "Сельскохозяйственная техника": "С/х техника",
    "Легковые автомобили": "Легковые авто",
    "Части и принадлежности для автомобилей и тракторов": "Автозапчасти",
    "Мебель, включая медицинскую": "Мебель",
    "Лекарственные средства, расфасованные для розничной продажи": "Лекарства",
    "Говядина": "Говядина",
    "Мясо и пищевые субпродукты домашней птицы": "Мясо птицы",
    "Молоко и сливки несгущенные": "Молоко (несгущ.)",
    "Молоко и сливки сгущенные и сухие": "Молоко (сгущ./сухое)",
    "Масло сливочное": "Масло сливочное",
    "Сыры и творог": "Сыры и творог",
    "Сахар": "Сахар",
    "Растительные масла": "Растительные масла",
    "Отходы, полученные при извлечении соевого масла": "Соевый жмых/шрот",
    "Каучук синтетический": "Синтетич. каучук",
    "Трубы из черных металлов": "Стальные трубы",
    "Металлоконструкции из черных металлов": "Металлоконструкции",
    "Двигатели внутреннего сгорания": "Двигатели ВС",
    "Вычислительные машины для автоматической обработки информации": "Компьютеры",
    "Машины и механические устройства специального назначения": "Спецмашины",
    "Аппаратура связи и части к ней": "Аппаратура связи",
}


def _to_number(x):
    """'1\xa0196,8' -> 1196.8; пустое/тире -> NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).replace("\xa0", "").replace(" ", "").replace(",", ".").strip()
    if s in {"", "-", "–", "—", "..."}:
        return np.nan
    return float(s)


def _split_name_unit(raw: str) -> tuple[str, str]:
    s = re.sub(r"\s+", " ", str(raw).replace("\n", " ")).strip()
    name, unit = s.rsplit(",", 1)
    return name.strip(), unit.strip()


def load_flow(flow: str) -> pd.DataFrame:
    """Возвращает tidy-таблицу: flow, product, unit, year, value."""
    df = pd.read_excel(FILES[flow], header=None, sheet_name=0)
    header_row = df.index[df.iloc[:, 1:].apply(lambda r: r.eq(2000).any(), axis=1)][0]
    years = df.iloc[header_row, 1:].dropna().astype(int).tolist()
    body = df.iloc[header_row + 1 :, : 1 + len(years)].dropna(subset=[0])
    body.columns = ["label", *years]

    records = []
    for _, row in body.iterrows():
        name, unit = _split_name_unit(row["label"])
        for y in years:
            records.append((flow, name, unit, y, _to_number(row[y])))
    tidy = pd.DataFrame(records, columns=["flow", "product", "unit", "year", "value"])
    tidy["product_short"] = tidy["product"].map(SHORT_NAMES).fillna(tidy["product"])
    return tidy


def load_all(save: bool = True) -> pd.DataFrame:
    tidy = pd.concat([load_flow("export"), load_flow("import")], ignore_index=True)
    # Экспорт легковых авто дан в штуках, импорт - в тыс. шт.: приводим к тыс. шт.
    mask = (tidy["flow"] == "export") & (tidy["product"] == "Легковые автомобили")
    tidy.loc[mask, "value"] = tidy.loc[mask, "value"] / 1000
    tidy.loc[mask, "unit"] = "тыс. шт."
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        tidy.to_csv(PROCESSED / "trade_tidy.csv", index=False, encoding="utf-8-sig")
    return tidy


if __name__ == "__main__":
    t = load_all()
    print(t.groupby("flow")["product"].nunique())
    print(t.isna().sum())
    print(t.head())
