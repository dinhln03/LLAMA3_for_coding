# The MIT License (MIT)

# Copyright (c) 2015 Yanzheng Li

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

## -----------------------------------------------------------------------------

def test_assert_true():
    try:
        assert True
        assert True, 'I want to believe.'
    except AssertionError:
        print 'This should not happen'

## -----------------------------------------------------------------------------

def test_assert_false():
    try:
        assert False
    except AssertionError:
        print 'I cannot believe'

## -----------------------------------------------------------------------------

def test_assert_on_truthy_exprs():
    try:
        assert 1
        assert 1 + 1
        assert 3.14 - 3.12
        assert not False
    except AssertionError:
        print 'This should not happen'

## -----------------------------------------------------------------------------

def test_assert_on_falsy_exprs():
    try:
        assert 0
    except AssertionError:
        print 'I cannot believe'

    try:
        assert 0 - 1
    except AssertionError:
        print 'I cannot believe'

    try:
        assert not True
    except AssertionError:
        print 'I cannot believe'

    try:
        assert 3.12 - 3.14
    except AssertionError:
        print 'I cannot believe'

## -----------------------------------------------------------------------------

test_assert_true()
test_assert_false()
test_assert_on_truthy_exprs()
test_assert_on_falsy_exprs()

## -----------------------------------------------------------------------------
