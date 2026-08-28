from pathlib import Path

import openpyxl
import pandas as pd

from aptafind.data.harmonization_schema import (
    SourceMeasurement,
    build_harmonized_tables,
    normalize_source_sequence,
)
from aptafind.data.source_adapters import (
    adapt_aptabench,
    adapt_n2a2,
    adapt_xiao_itc,
    adapt_xiao_specificity,
)


def test_sequence_normalization_removes_only_presentation_markers() -> None:
    assert normalize_source_sequence("5' AC GT 3'", "ssDNA") == (
        "ACGT",
        "valid_dna",
    )
    assert normalize_source_sequence("/5Phos/ACGT", "DNA") == (
        None,
        "unsupported_or_modified_sequence",
    )


def test_harmonized_tables_preserve_labels_and_build_model_views() -> None:
    shared = {
        "source_dataset": "fixture",
        "source_file": "fixture.csv",
        "source_sheet": None,
        "source_aptamer_id": "A1",
        "sequence_raw": "ACGTACGT",
        "polymer_type": "DNA",
        "sequence_role": "published_sequence",
        "source_target_id": None,
        "target_name_raw": "example",
        "target_smiles_raw": "CCO",
        "publication_id": "doi:10.0000/example",
        "measurement_type": "binary_binding",
        "measurement_value": 1.0,
        "measurement_unit": "binary",
        "evidence_type": "measured_binding",
        "is_target_measurement": True,
    }
    records = [
        SourceMeasurement(
            **shared,
            source_row=2,
            source_record_id="positive",
            binding_label="positive",
        ),
        SourceMeasurement(
            **shared,
            source_row=3,
            source_record_id="negative",
            binding_label="negative",
        ),
    ]

    tables = build_harmonized_tables(records)

    assert len(tables.aptamers) == 1
    assert len(tables.targets) == 1
    assert len(tables.measurements) == 2
    assert set(tables.model_interactions["label"]) == {0, 1}
    assert tables.model_interactions["pair_label_conflict"].all()
    assert tables.generation_positive_pairs.empty


def test_aptabench_adapter_filters_rna_and_preserves_binary_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "aptabench.csv"
    pd.DataFrame(
        [
            {
                "type": "DNA",
                "sequence": "ACGT",
                "canonical_smiles": "CCO",
                "pKd_value": 6.0,
                "label": 1,
                "buffer": "buffer",
                "origin": "https://doi.org/10.0000/example",
                "source": "Manual",
            },
            {
                "type": "DNA",
                "sequence": "TGCA",
                "canonical_smiles": "CCN",
                "pKd_value": None,
                "label": 0,
                "buffer": "buffer",
                "origin": None,
                "source": "Specificity",
            },
            {
                "type": "RNA",
                "sequence": "ACGU",
                "canonical_smiles": "CCO",
                "pKd_value": 5.0,
                "label": 1,
                "buffer": None,
                "origin": None,
                "source": "Manual",
            },
        ]
    ).to_csv(path, index=False)

    result = adapt_aptabench(path, "aptabench")

    assert len(result.records) == 2
    assert {record.binding_label for record in result.records} == {
        "positive",
        "negative",
    }
    assert result.audit["dna_rows_emitted"] == 2


def test_n2a2_adapter_emits_five_target_measurements_per_sequence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "n2a2.xlsx"
    row = {"seq": "ACGT", "replicate": 3}
    for code in ("Kyn", "KA", "3HK", "3HA", "XA"):
        row[f"{code}_mean"] = 0.5
        row[f"{code}_stdev"] = 0.1
    row["3HK_mean"] = 3.0
    pd.DataFrame([row]).to_excel(path, sheet_name="mean_z_score_df", index=False)

    result = adapt_n2a2(path, "n2a2")

    assert len(result.records) == 5
    assert sum(record.binding_label == "screen_positive" for record in result.records) == 1
    assert all(record.replicate_count == 3 for record in result.records)


def test_xiao_specificity_adapter_handles_multiple_side_by_side_panels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "specificity.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Panels"
    sheet.append([None, "A1", "A2", None, "B1"])
    sheet.append(["Target A (Target)", 1, 1, "Target B (Target)", 1])
    sheet.append(["Off A", 0, 0.2, "Off B", 0])
    sheet.append(["Specificity Score", 1, 2, "Specificity Score", 3])
    workbook.save(path)

    result = adapt_xiao_specificity(path, "xiao")

    assert len(result.records) == 6
    assert result.audit == {
        "aptamer_identifiers": 3,
        "target_measurements": 3,
        "non_target_measurements": 3,
        "total_measurements": 6,
    }


def test_xiao_itc_adapter_preserves_thermodynamic_details(tmp_path: Path) -> None:
    path = tmp_path / "itc.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "All ITC Data"
    sheet.append(["Target A", None, None, None, None, None, None])
    sheet.append(
        ["Aptamer ID", "KD", "delta H", "delta S", "T delta S", "Sites", "Ref"]
    )
    sheet.append([None, "(nM)", "(kcal/mol)", "(cal/mol/K)", "(kcal/mol)"])
    sheet.append(["A1", 25, -10, -20, -6, 0.8, 1])
    workbook.save(path)

    result = adapt_xiao_itc(path, "xiao")

    assert len(result.records) == 1
    record = result.records[0]
    assert record.measurement_value == 25
    assert record.details["delta_h_bind_kcal_per_mol"] == -10
    assert result.audit["target_groups"] == 1
