@echo off
echo ================================================================================
echo COMPILING ARXIV / EARTHARXIV PREPRINT PDF
echo ================================================================================

where typst >nul 2>nul
if %errorlevel% equ 0 (
    echo Compiling via Typst...
    typst compile manuscript.md manuscript.pdf
    echo PDF generated at: manuscript.pdf
    goto done
)

where pandoc >nul 2>nul
if %errorlevel% equ 0 (
    echo Compiling via Pandoc...
    pandoc manuscript.md -o manuscript.pdf --pdf-engine=xelatex --toc -N
    echo PDF generated at: manuscript.pdf
    goto done
)

echo NEITHER PANDOC NOR TYPST FOUND IN PATH!
echo Please install Pandoc (https://pandoc.org) or Typst (https://typst.app) to build PDF automatically.
echo Or upload manuscript.md and figures/ directly to Overleaf / arXiv TeX compiler.

:done
pause
