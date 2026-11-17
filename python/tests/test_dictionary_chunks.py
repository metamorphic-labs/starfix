"""Dictionary chunks whose combined dictionary does not fit the index type (ITL-1178).

Each chunk is valid on its own, but ``combine_chunks`` (which keeps the index type) fails on
their union. The hash must be that of the decoded values in one chunk, through every call,
for int8, uint8 and int16 indices, disjoint and overlapping dictionaries, with nulls, at the top
level and inside a struct or a list. ``test_differential.py`` compares such inputs with
starfix-python (``_support.dictionary_table``).
"""

import pyarrow as pa
import pytest

from _support import dictionary_chunks, wrap_chunks
from starfix import ArrowDigester

# (index type, values per chunk, offset of the second chunk's values): disjoint when the offset
# is the chunk size, overlapping when it is smaller; the union exceeds the index capacity.
CASES = [
    (pa.int8(), 100, 100),
    (pa.int8(), 120, 60),
    (pa.uint8(), 200, 200),
    (pa.uint8(), 250, 100),
    (pa.int16(), 20_000, 20_000),
    (pa.int16(), 30_000, 10_000),
]


def _id(case):
    index, size, offset = case
    return f"{index}-{'disjoint' if offset == size else 'overlapping'}"


@pytest.fixture(params=CASES, ids=_id)
def chunks(request):
    index, size, offset = request.param
    encoded, decoded = dictionary_chunks(index, size, offset)
    with pytest.raises(pa.ArrowException):
        pa.chunked_array(encoded).combine_chunks()  # the case is beyond the index capacity
    return encoded, decoded


def test_a_chunked_array(chunks):
    encoded, decoded = chunks
    expected = ArrowDigester.hash_array(pa.concat_arrays(decoded))
    assert ArrowDigester.hash_array(pa.chunked_array(encoded)) == expected
    for options in ({"ordered_columns": True}, {"ordered_struct_fields": True}):
        assert ArrowDigester.hash_array(pa.chunked_array(encoded), **options) == ArrowDigester.hash_array(
            pa.concat_arrays(decoded), **options
        )


@pytest.mark.parametrize("kind", ["top", "struct", "list"])
def test_a_multi_chunk_table(chunks, kind):
    encoded, decoded = chunks
    encoded_chunks, decoded_chunks = wrap_chunks(encoded, kind), wrap_chunks(decoded, kind)
    table = pa.Table.from_batches([pa.RecordBatch.from_arrays([c], names=["c"]) for c in encoded_chunks])
    assert table.num_rows == sum(len(c) for c in decoded_chunks) and len(table.to_batches()) == 2
    one = pa.RecordBatch.from_arrays([pa.concat_arrays(decoded_chunks)], names=["c"])
    expected = ArrowDigester.hash_record_batch(one)
    assert ArrowDigester.hash_table(table) == expected
    assert ArrowDigester.prepare(table.schema).hash_table(table) == expected
    digester = ArrowDigester(table.schema)
    for batch in table.to_batches():
        digester.update(batch)
    assert digester.finalize() == expected
    assert ArrowDigester.hash_array(table.column("c")) == ArrowDigester.hash_array(pa.concat_arrays(decoded_chunks))
