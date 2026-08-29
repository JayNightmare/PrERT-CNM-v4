"""CNMv2 CLI shim (legacy entrypoint delegating to prert.cli.cnmv2)."""

from prert.cli.cnmv2 import main

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    raise SystemExit(main())
