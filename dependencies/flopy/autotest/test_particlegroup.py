import tempfile
from pathlib import Path

import numpy as np

from flopy.modpath import ParticleData, ParticleGroup


def test_pgroup_release_data():
    # create particles
    partlocs = []
    partids = []
    nrow = 21
    for i in range(nrow):
        partlocs.append((0, i, 2))
        partids.append(i)
    pdata = ParticleData(partlocs, structured=True, particleids=partids)
    pgrd1 = ParticleGroup(
        particlegroupname="PG1",
        particledata=pdata,
        filename="exrd1.sloc",
        releasedata=0.0,
    )
    nripg2 = 10
    ripg2 = 1.0
    pgrd2 = ParticleGroup(
        particlegroupname="PG2",
        particledata=pdata,
        filename="exrd2.sloc",
        releasedata=[nripg2, 0.0, ripg2],
    )
    nripg3 = 10
    pgrd3 = ParticleGroup(
        particlegroupname="PG3",
        particledata=pdata,
        filename="exrd3.sloc",
        releasedata=[nripg3, np.arange(0, nripg3)],
    )

    assert len(pgrd1.releasetimes) == 1, (
        f"mp7: pgroup with releaseoption 1 returned "
        f"len(releasetimes)={len(pgrd1.releasetimes)}. Should be 1"
    )
    assert len(pgrd2.releasetimes) == nripg2, (
        f"mp7: pgroup with releaseoption 2 returned "
        f"len(releasetimes)={len(pgrd2.releasetimes)}. Should be {nripg2}"
    )
    assert type(pgrd2.releaseinterval) == type(ripg2), (
        f"mp7: pgroup with releaseoption 2 returned "
        f"type(releaseinterval)={type(pgrd2.releaseinterval)}. "
        f"Should remain as {type(ripg2)}"
    )
    assert len(pgrd3.releasetimes) == nripg3, (
        f"mp7: pgroup with releaseoption 3 returned "
        f"len(releasetimes)={len(pgrd3.releasetimes)}. Should be {nripg3}"
    )


def test_pgroup_write_with_release_timing():
    """Test that particle groups can be written to files with all release options."""
    # create particles
    partlocs = []
    partids = []
    nrow = 21
    for i in range(nrow):
        partlocs.append((0, i, 2))
        partids.append(i)
    pdata = ParticleData(partlocs, structured=True, particleids=partids)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test releaseoption 1 (single release time)
        pgrd1 = ParticleGroup(
            particlegroupname="PG1",
            particledata=pdata,
            filename=None,  # Write internally to test that path
            releasedata=0.0,
        )

        # Test releaseoption 2 (time range with interval)
        nripg2 = 10
        ripg2 = 1.0
        pgrd2 = ParticleGroup(
            particlegroupname="PG2",
            particledata=pdata,
            filename=None,
            releasedata=[nripg2, 0.0, ripg2],
        )

        # Test releaseoption 3 (explicit list of times).
        # this triggers Util2d with None model
        nripg3 = 7
        pgrd3 = ParticleGroup(
            particlegroupname="PG3",
            particledata=pdata,
            filename=None,
            releasedata=[nripg3, np.array([0.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])],
        )

        # Write each particle group to test all release options
        for pg in [pgrd1, pgrd2, pgrd3]:
            sim_file = tmpdir / f"{pg.particlegroupname}_test.sim"
            with open(sim_file, "w") as f:
                # This should not raise an AttributeError
                pg.write(f, ws=str(tmpdir))

            # Verify file was created and has content
            assert sim_file.exists()
            assert sim_file.stat().st_size > 0
