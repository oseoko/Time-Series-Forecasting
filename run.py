"""
한 방 파이프라인 — dataset → knowledge → experience → evaluate → HTML 리포트.

covariate는 --channels로 선택 (프리셋 all/sig4/sig2 또는 쉼표구분).
dataset·knowledge는 이미 있으면 건너뛴다. 산출물은 memory/run_<tag>/,
리포트는 analysis/fs_report_<tag>.html.

기본값(sig2 + ±20%)이 ablation에서 가장 좋았던 설정이다. README 참고.

실행:
  python run.py                                    # sig2, ±20% (권장 설정)
  python run.py --channels all                     # 23채널 전부
  python run.py --scale-pct 0                      # 보정 크기 제한 없음
  python run.py --regen-knowledge                  # STEP1부터 새로
"""
import argparse
import glob
import os
import subprocess
import sys

from config import MEMDIR, resolve_channels, run_tag

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(cmd, log=None):
    print(f">> {' '.join(cmd)}", flush=True)
    with open(log, "w") if log else sys.stdout as f:
        r = subprocess.run(cmd, cwd=HERE, stdout=(f if log else None),
                           stderr=subprocess.STDOUT if log else None)
    if r.returncode:
        sys.exit(f"[run] 실패 (exit {r.returncode}): {' '.join(cmd)}"
                 + (f"  로그: {log}" if log else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", default="sig2", help="all/sig4/sig2 또는 쉼표구분 채널명")
    ap.add_argument("--scale-pct", type=float, default=20.0,
                    help="보정 크기 상한 ±N퍼센트 (기본 20 — ablation 최적); 0=제한 없음")
    ap.add_argument("--knowledge", default="knowledge.json", help="memory/ 아래 knowledge 파일명")
    ap.add_argument("--regen-knowledge", action="store_true", help="knowledge를 새로 생성(STEP1부터)")
    ap.add_argument("--out", default=None, help="HTML 출력 경로(기본 analysis/fs_report_<tag>.html)")
    args = ap.parse_args()

    resolve_channels(args.channels)                    # 채널 스펙 검증(틀리면 여기서 멈춤)
    tag = run_tag(args.channels)
    rundir = os.path.join(MEMDIR, f"run_{tag}")
    exp = os.path.join(rundir, "experience.json")
    kpath = os.path.join(MEMDIR, args.knowledge)
    out = args.out or os.path.join(HERE, "analysis", f"fs_report_{tag}.html")
    L = os.path.join(rundir, "log");  os.makedirs(rundir, exist_ok=True)
    sc = ["--scale-pct", str(args.scale_pct)]
    kn = ["--knowledge", args.knowledge]

    # 1. dataset — parquet이 이미 있으면 건너뜀 (재구축은 dataset/raw의 원본 xlsx 필요)
    if os.path.exists(os.path.join(HERE, "dataset", "master_daily.parquet")):
        print("[1/5] dataset: 이미 있음 → skip")
    else:
        print("[1/5] dataset: build_dataset (dataset/raw의 원본 xlsx 필요)")
        sh(["python", "dataset/build_dataset.py"])

    # 2. knowledge — 채널 무관하게 23채널 전부 생성해두고 필요한 것만 골라 쓴다
    if args.regen_knowledge or not os.path.exists(kpath):
        print(f"[2/5] knowledge: 생성 → {kpath}")
        sh(["python", "knowledge.py", "--out", args.knowledge], log=L + "_knowledge.txt")
    else:
        print(f"[2/5] knowledge: 재사용 {kpath}")

    # 3. experience (train)
    print(f"[3/5] experience: {args.channels} → {exp}")
    sh(["python", "experience.py", "--channels", args.channels, "--exp", exp] + sc + kn,
       log=L + "_exp.txt")

    # 4. evaluate (test) + 사용자용 설명
    print(f"[4/5] evaluate: {args.channels} (+explain)")
    sh(["python", "evaluate.py", "--channels", args.channels, "--exp", exp, "--explain"] + sc + kn,
       log=L + "_eval.txt")

    # 5. HTML 리포트
    res = sorted(glob.glob(os.path.join(rundir, "test_results*.json")))
    if not res:
        sys.exit(f"[run] test_results 없음: {rundir}")
    print(f"[5/5] report → {out}")
    sh(["python", "analysis/report_page.py", "--res", res[-1], "--exp", exp,
        "--knowledge", kpath, "--out", out, "--standalone"], log=L + "_report.txt")

    print(f"\n✅ 완료 — 산출물: {rundir}/   리포트: {out}")


if __name__ == "__main__":
    main()
