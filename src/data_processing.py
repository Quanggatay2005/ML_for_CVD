"""
BRFSS Data Processing Pipeline -- PySpark Edition
=================================================
Handles:
  1. Data Cleaning      -- Drop high-missingness columns, replace BRFSS sentinel
                          codes (7/9/77/99/777/999) with null, drop rows missing target
  2. Feature Engineering -- Curated 30-column clinical selection, binarise CVDINFR4,
                           encode ordinal/categorical variables
  3. Sample Reduction   -- Stratified undersampling: keep 100% minority + 3x majority
                          (configurable ratio), preserving sufficient scale for
                          LightGBM, Random Forest, SVM, etc.

Output: data/new_brfss.csv  (clean, ML-ready, ~3-4x minority class rows)
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
import pandas as pd
import os

# ------------------------------------------------------------------------------
# Clinically-relevant feature set (328 raw BRFSS cols -> 30 selected)
# ------------------------------------------------------------------------------
TARGET_COL = "CVDINFR4"          # Ever told had heart attack (1=Yes, 2=No, 7/9=unknown)

FEATURE_COLS = [
    # --- Demographics ---
    "_AGEG5YR",   # Age group (5-year bands, 1-13)
    "SEXVAR",     # Sex (1=Male, 2=Female)
    "_IMPRACE",   # Imputed race/ethnicity (1-6)
    "_EDUCAG",    # Education level (1-4)
    "_INCOMG1",   # Income group (1-7)

    # --- General Health ---
    "GENHLTH",    # General health (1=Excellent ... 5=Poor)
    "PHYSHLTH",   # Physical health not good (days/month 0-30)
    "MENTHLTH",   # Mental health not good (days/month 0-30)
    "_HLTHPLN",   # Any health plan (1=Yes, 2=No)

    # --- Cardiovascular Risk ---
    "CVDCRHD4",   # Ever diagnosed with CHD (1=Yes, 2=No)
    "CVDSTRK3",   # Ever had stroke (1=Yes, 2=No)
    "TOLDHI3",    # Told high cholesterol (1=Yes, 2=No)
    "BPHIGH6",    # Told high blood pressure (1=Yes, 2=No, 4=Borderline)
    "BPMEDS",     # Currently on BP meds (1=Yes, 2=No)

    # --- Diabetes & BMI ---
    "DIABETE4",   # Diabetes diagnosis (1=Yes, 2=Prediabetes, 3=No, 4=Gestational)
    "_BMI5",      # Computed BMI x 100

    # --- Lifestyle ---
    "_TOTINDA",   # Any physical activity (1=Yes, 2=No)
    "SLEPTIM1",   # Sleep hours per night
    "SMOKE100",   # Smoked >=100 cigarettes ever (1=Yes, 2=No)
    "_RFSMOK3",   # Current smoker (1=No, 2=Yes)
    "_RFDRHV8",   # Heavy alcohol use (1=No, 2=Yes)

    # --- Comorbidities ---
    "CHCCOPD3",   # COPD / emphysema (1=Yes, 2=No)
    "ADDEPEV3",   # Depressive disorder (1=Yes, 2=No)
    "CHCKDNY2",   # Kidney disease (1=Yes, 2=No)
    "HAVARTH4",   # Arthritis (1=Yes, 2=No)
    "CHCOCNC1",   # Colon/rectum cancer (1=Yes, 2=No)

    # --- Preventive Care ---
    "CHECKUP1",   # Last routine checkup (1=< 1yr ... 4=>=5yr, 8=Never)
    "EXERANY2",   # Any exercise past 30 days (1=Yes, 2=No)
    "SLEPTIM1",   # (duplicate guard handled below at dedup step)
]

# Deduplicate feature list while preserving order
_seen = set()
FEATURE_COLS = [c for c in FEATURE_COLS if not (c in _seen or _seen.add(c))]

# BRFSS sentinel codes that mean "unknown / refused / N/A"
# These must be treated as missing, not as real numeric values
SENTINEL_SINGLE = {7, 9}           # 1-digit fields
SENTINEL_DOUBLE = {77, 99}         # 2-digit fields
SENTINEL_TRIPLE = {777, 999}       # 3-digit fields
SENTINEL_NONE   = {8, 88, 888}     # "None / not applicable" -- keep as 0 OR null per context
ALL_SENTINELS   = SENTINEL_SINGLE | SENTINEL_DOUBLE | SENTINEL_TRIPLE | SENTINEL_NONE


# ------------------------------------------------------------------------------
class BRFSSDataProcessor:
    """
    PySpark-based BRFSS preprocessing pipeline.

    Usage (generate new_brfss.csv):
        proc = BRFSSDataProcessor()
        proc.run_full_pipeline("data/brfss_data.csv", "data/new_brfss.csv")

    Legacy usage (returns pandas DataFrame for model training):
        df = proc.process_and_get_pandas("data/brfss_data.csv", "CVDINFR4")
    """

    def __init__(self, app_name: str = "BRFSS_CVD_Preprocessing",
                 driver_memory: str = "6g"):
        self.spark = (
            SparkSession.builder
            .appName(app_name)
            .config("spark.driver.memory", driver_memory)
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.sql.execution.arrow.pyspark.enabled", "false")
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("WARN")

    # --------------------------------------------------------------------------
    # PUBLIC INTERFACE
    # --------------------------------------------------------------------------

    def run_full_pipeline(
        self,
        input_path: str,
        output_path: str,
        target_col: str = TARGET_COL,
        majority_ratio: float = 3.0,
        missing_col_threshold: float = 0.50,
    ) -> None:
        """
        Full end-to-end PySpark pipeline:
          Load -> Clean -> Feature Engineer -> Impute -> Sample Reduce -> Export CSV

        Parameters
        ----------
        input_path           : Path to raw brfss_data.csv
        output_path          : Destination path for new_brfss.csv
        target_col           : Target column (default: CVDINFR4)
        majority_ratio       : Majority rows = minority_count x ratio  (default 3.0 -> 1:3 balance)
        missing_col_threshold: Drop columns missing more than this fraction (default 0.50)
        """
        print("\n" + "="*60)
        print("  BRFSS PySpark Pipeline -- Starting")
        print("="*60)

        df = self._load_data(input_path)
        df = self._drop_high_missing_cols(df, threshold=missing_col_threshold)
        df = self._select_features(df, target_col)
        df = self._clean_sentinels(df, target_col)
        df = self._feature_engineering(df, target_col)
        df = self._impute(df, target_col)
        df = self._sample_reduce(df, target_col, majority_ratio=majority_ratio)
        self._export_csv(df, output_path)

        print("\n" + "="*60)
        print(f"  Pipeline complete -> {output_path}")
        print("="*60 + "\n")

    def process_and_get_pandas(
        self,
        data_path: str,
        target_col: str = TARGET_COL,
    ) -> pd.DataFrame:
        """
        Legacy-compatible method.
        Runs the full pipeline and returns a Pandas DataFrame for model training.
        Does NOT write new_brfss.csv -- use run_full_pipeline() for that.
        """
        print("Loading data via PySpark (legacy mode)...")
        df = self._load_data(data_path)
        df = self._drop_high_missing_cols(df)
        df = self._select_features(df, target_col)
        df = self._clean_sentinels(df, target_col)
        df = self._feature_engineering(df, target_col)
        df = self._impute(df, target_col)
        df = self._sample_reduce(df, target_col, majority_ratio=3.0)

        print("\n--- Processed DataFrame: Schema ---")
        df.printSchema()
        print(f"\n--- First 5 rows ---")
        df.show(5, truncate=False)
        n_rows = df.count()
        print(f"Total rows: {n_rows:,}  |  Total columns: {len(df.columns)}")

        print("\nConverting to Pandas for model training...")
        return df.toPandas()

    # --------------------------------------------------------------------------
    # STEP 1 -- LOAD
    # --------------------------------------------------------------------------

    def _load_data(self, path: str) -> DataFrame:
        abs_path = os.path.abspath(path)
        print(f"\n[1/6] Loading: {abs_path}")
        df = self.spark.read.csv(abs_path, header=True, inferSchema=True)
        print(f"      Loaded {df.count():,} rows x {len(df.columns)} columns")
        return df

    # --------------------------------------------------------------------------
    # STEP 2 -- DROP HIGH-MISSINGNESS COLUMNS
    # --------------------------------------------------------------------------

    def _drop_high_missing_cols(
        self, df: DataFrame, threshold: float = 0.50
    ) -> DataFrame:
        print(f"\n[2/6] Dropping columns with >{threshold*100:.0f}% missing values...")
        n = df.count()
        null_fracs = {
            c: df.filter(F.col(c).isNull()).count() / n
            for c in df.columns
        }
        drop_cols = [c for c, frac in null_fracs.items() if frac > threshold]
        df = df.drop(*drop_cols)
        print(f"      Dropped {len(drop_cols)} columns -- {len(df.columns)} remaining")
        return df

    # --------------------------------------------------------------------------
    # STEP 3 -- SELECT CURATED FEATURE COLUMNS
    # --------------------------------------------------------------------------

    def _select_features(self, df: DataFrame, target_col: str) -> DataFrame:
        print(f"\n[3a] Selecting curated feature set + target '{target_col}'...")
        available = set(df.columns)

        # Keep only features that actually exist in the dataset
        keep = [c for c in FEATURE_COLS if c in available]
        missing_req = [c for c in FEATURE_COLS if c not in available]
        if missing_req:
            print(f"      [WARNING] Features not found in dataset (skipped): {missing_req}")

        # Always keep the target
        if target_col not in keep:
            keep = [target_col] + keep

        df = df.select(*keep)
        print(f"      Selected {len(df.columns)} columns ({len(keep)-1} features + 1 target)")
        return df

    # --------------------------------------------------------------------------
    # STEP 4 -- CLEAN SENTINEL CODES & BINARISE TARGET
    # --------------------------------------------------------------------------

    def _clean_sentinels(self, df: DataFrame, target_col: str) -> DataFrame:
        """
        Replace BRFSS sentinel codes with null for all numeric feature columns.
        BRFSS uses specific values to encode 'Don't know', 'Refused', 'Not asked':
          - Single digit fields  : 7 = Don't know, 9 = Refused/Missing
          - Two digit fields     : 77, 99
          - Three digit fields   : 777, 999
          - None/N-A fields      : 8, 88, 888
        """
        print(f"\n[3b] Replacing BRFSS sentinel codes with null...")
        feature_cols = [c for c in df.columns if c != target_col]
        sentinel_list = sorted(ALL_SENTINELS)

        for col_name in feature_cols:
            # Only replace in numeric integer-like columns (not continuous like _BMI5)
            col_type = dict(df.dtypes)[col_name]
            if col_type in ("int", "bigint", "smallint", "tinyint", "integer"):
                df = df.withColumn(
                    col_name,
                    F.when(F.col(col_name).isin(sentinel_list), None)
                     .otherwise(F.col(col_name))
                )

        # Handle target: CVDINFR4 (1=Yes, 2=No, 7=DK, 9=Refused) -> drop unknowns
        df = df.filter(F.col(target_col).isNotNull())
        df = df.filter(~F.col(target_col).isin([7, 9]))

        remaining = df.count()
        print(f"      {remaining:,} rows remain after sentinel/target cleaning")
        return df

    # --------------------------------------------------------------------------
    # STEP 5 -- FEATURE ENGINEERING
    # --------------------------------------------------------------------------

    def _feature_engineering(self, df: DataFrame, target_col: str) -> DataFrame:
        """
        Feature transformations:
          - Binarise target: CVDINFR4 == 1 -> 1, else -> 0
          - Cast all feature columns to Double for downstream ML
          - Derive binary flags from ordinal columns (e.g., poor health, obese, etc.)
        """
        print(f"\n[4/6] Feature engineering...")

        # -- Binarise target --------------------------------------------------
        # CVDINFR4: 1 = "Yes, ever had heart attack" -> 1
        #           2 = "No"                          -> 0
        df = df.withColumn(
            target_col,
            F.when(F.col(target_col) == 1, 1).otherwise(0).cast(DoubleType())
        )

        # -- Derived / recoded features ---------------------------------------
        # Poor general health flag (GENHLTH 4-5 = Very poor / Poor)
        if "GENHLTH" in df.columns:
            df = df.withColumn(
                "POOR_HEALTH",
                F.when(F.col("GENHLTH").isin([4, 5]), 1).otherwise(0).cast(DoubleType())
            )

        # Current smoker flag
        if "_RFSMOK3" in df.columns:
            df = df.withColumn(
                "SMOKER",
                F.when(F.col("_RFSMOK3") == 2, 1).otherwise(0).cast(DoubleType())
            )

        # Physically inactive flag
        if "_TOTINDA" in df.columns:
            df = df.withColumn(
                "INACTIVE",
                F.when(F.col("_TOTINDA") == 2, 1).otherwise(0).cast(DoubleType())
            )

        # Short sleep flag (< 6 hours)
        if "SLEPTIM1" in df.columns:
            df = df.withColumn(
                "SHORT_SLEEP",
                F.when(F.col("SLEPTIM1") < 6, 1).otherwise(0).cast(DoubleType())
            )

        # High blood pressure flag
        if "BPHIGH6" in df.columns:
            df = df.withColumn(
                "HYPERTENSION",
                F.when(F.col("BPHIGH6") == 1, 1).otherwise(0).cast(DoubleType())
            )

        # -- Cast all features to Double --------------------------------------
        feature_cols = [c for c in df.columns if c != target_col]
        for c in feature_cols:
            df = df.withColumn(c, F.col(c).cast(DoubleType()))

        print(f"      Total columns after engineering: {len(df.columns)}")
        pos = df.filter(F.col(target_col) == 1).count()
        neg = df.filter(F.col(target_col) == 0).count()
        total = pos + neg
        print(f"      Class distribution -- CVDINFR4=1: {pos:,} ({pos/total*100:.1f}%)  "
              f"| CVDINFR4=0: {neg:,} ({neg/total*100:.1f}%)")
        return df

    # --------------------------------------------------------------------------
    # STEP 6 -- MEDIAN IMPUTATION
    # --------------------------------------------------------------------------

    def _impute(self, df: DataFrame, target_col: str) -> DataFrame:
        print(f"\n[5/6] Imputing missing values (median strategy)...")
        feature_cols = [c for c in df.columns if c != target_col]

        # Only impute columns that actually have nulls
        null_counts = df.select(
            [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in feature_cols]
        ).collect()[0].asDict()

        cols_to_impute = [c for c in feature_cols if null_counts[c] > 0]
        output_cols    = [c + "_imp" for c in cols_to_impute]

        if cols_to_impute:
            imputer = Imputer(
                inputCols=cols_to_impute,
                outputCols=output_cols,
                strategy="median"
            )
            model = imputer.fit(df)
            df = model.transform(df)

            # Replace original columns with imputed versions
            for orig, imp in zip(cols_to_impute, output_cols):
                df = df.drop(orig).withColumnRenamed(imp, orig)

            print(f"      Imputed {len(cols_to_impute)} columns")
        else:
            print("      No null values found -- skipping imputation")

        return df

    # --------------------------------------------------------------------------
    # STEP 7 -- STRATIFIED SAMPLE REDUCTION
    # --------------------------------------------------------------------------

    def _sample_reduce(
        self,
        df: DataFrame,
        target_col: str,
        majority_ratio: float = 3.0,
        seed: int = 42,
    ) -> DataFrame:
        """
        Stratified undersampling to address class imbalance while preserving scale.
        Strategy:
          - Keep ALL minority class rows (CVDINFR4 = 1)
          - Undersample majority class to minority_count x majority_ratio
          - Result: 1:3 minority:majority ratio by default
        """
        print(f"\n[6/6] Stratified sample reduction (1:{majority_ratio:.0f} balance)...")

        minority = df.filter(F.col(target_col) == 1)
        majority = df.filter(F.col(target_col) == 0)

        n_minority = minority.count()
        n_majority = majority.count()
        n_target   = int(n_minority * majority_ratio)

        print(f"      Minority (=1): {n_minority:,} rows  |  Majority (=0): {n_majority:,} rows")

        if n_majority > n_target:
            sample_fraction = n_target / n_majority
            majority_sampled = majority.sample(
                withReplacement=False,
                fraction=sample_fraction,
                seed=seed
            )
        else:
            majority_sampled = majority
            print("      Majority class is already smaller than target -- keeping all")

        df_balanced = minority.union(majority_sampled).orderBy(F.rand(seed=seed))

        n_final = df_balanced.count()
        pos = df_balanced.filter(F.col(target_col) == 1).count()
        neg = n_final - pos
        print(f"      Final dataset: {n_final:,} rows  "
              f"(pos={pos:,} | neg={neg:,}  ratio 1:{neg/pos:.1f})")
        return df_balanced

    # --------------------------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------------------------

    def _export_csv(self, df: DataFrame, output_path: str) -> None:
        """
        Write the processed Spark DataFrame to a single CSV file.
        Uses toPandas().to_csv() to avoid the HADOOP_HOME / winutils.exe
        requirement that Spark's native CSV writer imposes on Windows.
        """
        abs_out = os.path.abspath(output_path)
        print(f"\n      Converting to Pandas for CSV export...")
        pandas_df = df.toPandas()
        print(f"      Writing CSV -> {abs_out}  ({len(pandas_df):,} rows x {len(pandas_df.columns)} cols)")
        pandas_df.to_csv(abs_out, index=False, encoding="utf-8")
        print(f"      [OK] Saved: {abs_out}")

    # --------------------------------------------------------------------------
    # TEARDOWN
    # --------------------------------------------------------------------------

    def stop(self) -> None:
        """Stop the underlying SparkSession."""
        self.spark.stop()
        print("SparkSession stopped.")


# ------------------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    input_csv  = sys.argv[1] if len(sys.argv) > 1 else "data/brfss_data.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "data/new_brfss.csv"

    processor = BRFSSDataProcessor()
    try:
        processor.run_full_pipeline(
            input_path=input_csv,
            output_path=output_csv,
            target_col=TARGET_COL,
            majority_ratio=3.0,
        )
    finally:
        processor.stop()
