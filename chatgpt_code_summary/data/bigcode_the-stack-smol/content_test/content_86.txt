#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# python version 2.7.6

import magic

mime = magic.Magic(mime=True)

print mime.from_file("/Users/mac/Documents/data/fastq/8.fastq")