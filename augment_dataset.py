import pandas as pd
import os
import shutil


DATASET_PATH = "data/dataset.csv"
BACKUP_PATH = "data/dataset_backup_before_augmentation.csv"


# ============================================================
# ADDITIONAL POLITICAL SENTIMENT DATA
# 40 POSITIVE + 40 NEGATIVE + 40 NEUTRAL
# ============================================================

additional_data = [

    # ========================================================
    # POSITIVE
    # ========================================================

    ("The government created new employment opportunities", "positive"),
    ("The employment generation program has been successful", "positive"),
    ("People appreciate the government's efforts to create jobs", "positive"),
    ("The new policy has improved employment opportunities", "positive"),
    ("The administration is doing excellent work", "positive"),
    ("The government has performed very well", "positive"),
    ("The administration has delivered strong results", "positive"),
    ("The new reforms have produced positive results", "positive"),
    ("The policy has benefited many citizens", "positive"),
    ("The government has made significant progress", "positive"),

    ("The healthcare reforms have improved public services", "positive"),
    ("Government hospitals have shown considerable improvement", "positive"),
    ("The education reforms are producing good results", "positive"),
    ("Students have benefited from the new education policy", "positive"),
    ("The infrastructure development is impressive", "positive"),
    ("Road infrastructure has improved significantly", "positive"),
    ("The administration handled the crisis effectively", "positive"),
    ("The government responded effectively to public concerns", "positive"),
    ("The new scheme has helped many families", "positive"),
    ("Citizens are satisfied with the new policy", "positive"),

    ("The government is doing a great job", "positive"),
    ("The administration is doing good work", "positive"),
    ("The policy is highly appreciated", "positive"),
    ("The reforms are highly successful", "positive"),
    ("The government's performance is excellent", "positive"),
    ("The new program is very effective", "positive"),
    ("The policy has generated positive outcomes", "positive"),
    ("The government has successfully reduced unemployment", "positive"),
    ("Employment levels have improved significantly", "positive"),
    ("The administration has created more job opportunities", "positive"),

    ("People are happy with the government's performance", "positive"),
    ("Citizens appreciate the new reforms", "positive"),
    ("The government has improved public transportation", "positive"),
    ("The healthcare system has become more efficient", "positive"),
    ("The education system has shown strong improvement", "positive"),
    ("The new infrastructure projects are beneficial", "positive"),
    ("The economic reforms have encouraged growth", "positive"),
    ("The government's economic policies are effective", "positive"),
    ("Public services have improved under the new administration", "positive"),
    ("The policy has received widespread appreciation", "positive"),


    # ========================================================
    # NEGATIVE
    # ========================================================

    ("Jobs are hard to find", "negative"),
    ("People are struggling to find employment", "negative"),
    ("There are not enough jobs for young people", "negative"),
    ("Employment opportunities are limited", "negative"),
    ("People cannot find suitable employment", "negative"),
    ("The unemployment situation is getting worse", "negative"),
    ("Young people are frustrated by the lack of jobs", "negative"),
    ("The government has failed to create enough jobs", "negative"),
    ("Job creation has been disappointing", "negative"),
    ("The employment situation remains poor", "negative"),

    ("People are frustrated with inflation", "negative"),
    ("Rising prices are hurting ordinary citizens", "negative"),
    ("Inflation has increased the cost of living", "negative"),
    ("The price increase has affected families", "negative"),
    ("High inflation is creating serious problems", "negative"),
    ("People are struggling because of rising prices", "negative"),
    ("The cost of essential goods is too high", "negative"),
    ("The government has failed to control inflation", "negative"),
    ("Inflation remains a major concern", "negative"),
    ("Citizens are unhappy with rising prices", "negative"),

    ("The government is doing a poor job", "negative"),
    ("The administration has performed badly", "negative"),
    ("The policy has failed to deliver results", "negative"),
    ("The government has failed to address unemployment", "negative"),
    ("The administration has disappointed citizens", "negative"),
    ("People are frustrated with the government's performance", "negative"),
    ("The reforms have not produced good results", "negative"),
    ("The policy has been ineffective", "negative"),
    ("The administration has failed to meet expectations", "negative"),
    ("The government's performance has been poor", "negative"),

    ("Corruption remains a serious problem", "negative"),
    ("People are concerned about corruption", "negative"),
    ("Poor governance has affected public services", "negative"),
    ("The healthcare system is facing serious problems", "negative"),
    ("Public hospitals are struggling to provide services", "negative"),
    ("The education system needs major improvement", "negative"),
    ("Infrastructure development has been disappointing", "negative"),
    ("Citizens are unhappy with public services", "negative"),
    ("The administration has not handled the situation effectively", "negative"),
    ("This decision is not good", "negative"),


    # ========================================================
    # NEUTRAL
    # ========================================================

    ("The parliament discussed the new policy", "neutral"),
    ("The government announced a new policy", "neutral"),
    ("The administration released a statement", "neutral"),
    ("The minister addressed the parliament", "neutral"),
    ("The government introduced a new scheme", "neutral"),
    ("Parliament held a meeting yesterday", "neutral"),
    ("The minister presented the proposal", "neutral"),
    ("The government published the report", "neutral"),
    ("Officials attended the meeting", "neutral"),
    ("The committee discussed the proposal", "neutral"),

    ("The government announced changes to the policy", "neutral"),
    ("The parliament passed the bill", "neutral"),
    ("The minister submitted the proposal", "neutral"),
    ("The administration released new guidelines", "neutral"),
    ("Officials conducted a review", "neutral"),
    ("The government organized a public meeting", "neutral"),
    ("The committee released its findings", "neutral"),
    ("The minister spoke about the new program", "neutral"),
    ("The government presented its budget", "neutral"),
    ("Parliament discussed the annual report", "neutral"),

    ("The policy was announced yesterday", "neutral"),
    ("The new scheme was introduced last month", "neutral"),
    ("The government held a press conference", "neutral"),
    ("The minister visited the district", "neutral"),
    ("Officials reported the latest figures", "neutral"),
    ("The committee met to discuss the issue", "neutral"),
    ("The government released employment data", "neutral"),
    ("The administration published the economic report", "neutral"),
    ("The parliament considered the proposal", "neutral"),
    ("The ministry released new information", "neutral"),

    ("The election commission released the schedule", "neutral"),
    ("The government issued a notification", "neutral"),
    ("The minister attended the conference", "neutral"),
    ("Officials reviewed the project", "neutral"),
    ("The administration submitted the report", "neutral"),
    ("Parliament debated the proposed legislation", "neutral"),
    ("The government announced the meeting date", "neutral"),
    ("The ministry published the statistics", "neutral"),
    ("The committee submitted its report", "neutral"),
    ("The government held discussions with officials", "neutral"),
]


# ============================================================
# LOAD EXISTING DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"❌ Dataset not found: {DATASET_PATH}"
    )


df = pd.read_csv(
    DATASET_PATH
)


df.columns = df.columns.str.strip()


required_columns = [
    "text",
    "label"
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"❌ Missing column: {column}"
        )


print("\n" + "=" * 65)
print("🗳️ POLITICAL SENTIMENT DATASET AUGMENTATION")
print("=" * 65)


print(
    f"\n📊 Existing dataset: {len(df)} rows"
)


print(
    "\nExisting distribution:"
)

print(
    df["label"].value_counts()
)


# ============================================================
# BACKUP
# ============================================================

shutil.copy2(
    DATASET_PATH,
    BACKUP_PATH
)


print(
    f"\n💾 Backup created:"
)

print(
    BACKUP_PATH
)


# ============================================================
# CREATE NEW DATAFRAME
# ============================================================

start_id = 1


if "id" in df.columns:

    numeric_ids = pd.to_numeric(
        df["id"],
        errors="coerce"
    )

    if numeric_ids.notna().any():

        start_id = int(
            numeric_ids.max()
        ) + 1


new_rows = []


for text, label in additional_data:

    row = {}

    if "id" in df.columns:

        row["id"] = start_id

        start_id += 1


    if "topic" in df.columns:

        row["topic"] = "politics"


    row["label"] = label

    row["text"] = text

    new_rows.append(row)


new_df = pd.DataFrame(
    new_rows
)


# ============================================================
# COMBINE
# ============================================================

combined = pd.concat(
    [
        df,
        new_df
    ],
    ignore_index=True
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

combined["text"] = (
    combined["text"]
    .astype(str)
    .str.strip()
)


before = len(combined)


combined = combined.drop_duplicates(
    subset=["text"],
    keep="first"
)


removed = (
    before -
    len(combined)
)


# ============================================================
# RESET IDS
# ============================================================

if "id" in combined.columns:

    combined["id"] = range(
        1,
        len(combined) + 1
    )


# ============================================================
# SAVE
# ============================================================

combined.to_csv(
    DATASET_PATH,
    index=False
)


# ============================================================
# RESULT
# ============================================================

print(
    "\n✅ Dataset augmentation completed!"
)


print(
    f"\nOriginal rows: {len(df)}"
)

print(
    f"New examples added: {len(new_df)}"
)

print(
    f"Duplicate examples removed: {removed}"
)

print(
    f"Final dataset size: {len(combined)}"
)


print(
    "\n📊 Final Class Distribution:"
)


print(
    combined["label"].value_counts()
)


print("\n" + "=" * 65)

print(
    "🚀 Next step:"
)

print(
    "python train.py"
)

print("=" * 65)