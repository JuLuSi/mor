#!/bin/bash
git push origin --delete gh-pages
git branch -D gh-pages
git checkout --orphan gh-pages
git rm -rf .
git checkout master docs mor
pushd docs
make html
cp -r build/html/. ../
popd
rm -rf docs mor
git add -A
git commit -m "rebuilt docs"
git push origin gh-pages
git checkout master
