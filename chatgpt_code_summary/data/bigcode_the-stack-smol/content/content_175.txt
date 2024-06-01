# Copyright 2021 Edoardo Riggio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Complexity: O(nlog(n))

def search_in_sorted_matrix(A, x):
    for S in A:
        if binary_search(S, x):
            return True

    return False


def binary_search(A, x):
    low = 0
    high = len(A) - 1
    mid = 0

    while low <= high:
        mid = (high + low) // 2

        if A[mid] < x:
            low = mid + 1
        elif A[mid] > x:
            high = mid - 1
        else:
            return True

    return False


mat = [[1, 2, 3, 4, 5], [9, 10, 20, 32, 55]]
print(search_in_sorted_matrix(mat, 56))
