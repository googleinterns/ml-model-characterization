# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers

class EmbeddingLookupSparseOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsEmbeddingLookupSparseOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = EmbeddingLookupSparseOptions()
        x.Init(buf, n + offset)
        return x

    # EmbeddingLookupSparseOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # EmbeddingLookupSparseOptions
    def Combiner(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def EmbeddingLookupSparseOptionsStart(builder): builder.StartObject(1)
def EmbeddingLookupSparseOptionsAddCombiner(builder, combiner): builder.PrependInt8Slot(0, combiner, 0)
def EmbeddingLookupSparseOptionsEnd(builder): return builder.EndObject()
