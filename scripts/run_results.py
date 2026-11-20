"""Regenerate the result files in results/. Run from the repository root:
    python scripts/run_results.py            # every part
    python scripts/run_results.py pan pet    # selected parts
Numbers are written to six significant figures; all random draws use fixed seeds, so a rerun reproduces every file.
The simulation tests of Section 5.4 are run separately by scripts/synthetic_validation.py."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import october          # noqa: E402  model checks of Sections 2-3 (Lemma 1, A5, Bayes factors, kernels, forms, coupling)
import conditioning     # noqa: E402
import drying_parts     # noqa: E402
import finite           # noqa: E402
import raman_parts      # noqa: E402
import records          # noqa: E402

PARTS = {
    # Sections 2-3 and S2-S3, S6-S7
    'lemma1': october.part_lemma1, 'a5': october.part_a5, 'a5cond': conditioning.part_a5cond,
    'bayes': october.part_bayes, 'kernel': october.part_kernel, 'prop3': october.part_prop3,
    'finite': finite.part_finite, 'forms': october.part_forms, 'forms_sampling': conditioning.part_forms_sampling,
    'route4_excess': conditioning.part_route4_excess, 'p4_bias': conditioning.part_p4_bias, 'gas_side': conditioning.part_gas_side, 'biot': conditioning.part_biot,
    'coupling': october.part_coupling, 'composition': october.part_composition,
    # polystyrene-chloroform model, Sections 4.1-4.2, S4-S5, S8.5
    'flat': drying_parts.part_flat, 'drying': drying_parts.part_drying, 'drying_checks': drying_parts.part_drying_checks,
    'skinfront': drying_parts.part_skinfront, 'remark3': drying_parts.part_remark3, 'clockspan': drying_parts.part_clockspan,
    'fastevap': drying_parts.part_fastevap,
    # records, Section 5 and S8
    'records': records.part_records, 'psunc': records.part_psunc, 'pet': records.part_pet,
    'channels': records.part_channels, 'pan': records.part_pan,
    'raman': raman_parts.part_raman, 'raman_check': raman_parts.part_raman_check,
}

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    unknown = [p for p in sys.argv[1:] if p not in PARTS]
    if unknown:
        sys.exit('unknown part(s) %s; valid parts: %s' % (', '.join(unknown), ', '.join(PARTS)))
    for name in sys.argv[1:] or PARTS:
        t0 = time.time()
        PARTS[name]()
        print('part %s done in %.0f s' % (name, time.time() - t0), flush=True)
