import onnx
import numpy as np
from collections import defaultdict
import sys
def advanced_onnx_analysis(model_path):
    model = onnx.load(model_path)
    initializers = model.graph.initializer
    
    total_stats = {
        'total_params': 0,
        'total_size_bytes': 0,
        'by_dtype': defaultdict(lambda: {'params': 0, 'size': 0}),
        'by_layer': defaultdict(lambda: {'params': 0, 'size': 0})
    }
    
    param_details = []
    
    for init in initializers:
        shape = init.dims
        dtype = init.data_type
        name = init.name
        param_count = np.prod(shape) if shape else 1
        
        dtype_size = get_dtype_size(dtype)
        size_bytes = param_count * dtype_size
        
        # 更新统计信息
        total_stats['total_params'] += param_count
        total_stats['total_size_bytes'] += size_bytes
        total_stats['by_dtype'][dtype]['params'] += param_count
        total_stats['by_dtype'][dtype]['size'] += size_bytes
        
        # 按层名统计（假设层名以'.'分隔）
        layer_name = '.'.join(name.split('.')[:-1]) if '.' in name else 'other'
        total_stats['by_layer'][layer_name]['params'] += param_count
        total_stats['by_layer'][layer_name]['size'] += size_bytes
        
        param_details.append({
            'name': name, 'shape': shape, 'dtype': dtype,
            'param_count': param_count, 'size_bytes': size_bytes
        })
    
    # 打印结果
    print(f"Advanced Analysis of: {model_path}")
    print("=" * 80)
    
    # 按参数数量排序
    param_details.sort(key=lambda x: x['param_count'], reverse=True)
    
    print("\nTop Largest Parameters:")
    print("-" * 80)
    for i, param in enumerate(param_details):
        print(f"{i+1:2d}. {param['name'][:40]:40} {str(param['shape']):20} {str(get_dtype_string(param['dtype'])):10}"
              f"{param['param_count']:12,} params  {format_size(param['size_bytes'])}")
    
    # 按数据类型统计
    print(f"\nBy Data Type:")
    print("-" * 80)
    for dtype, stats in total_stats['by_dtype'].items():
        dtype_name = onnx.TensorProto.DataType.Name(dtype) if hasattr(onnx.TensorProto.DataType, 'Name') else f"TYPE_{dtype}"
        print(f"{dtype_name:15} {stats['params']:12,} params  {format_size(stats['size'])}")
    
    # 汇总
    print(f"\nSUMMARY:")
    print("-" * 80)
    print(f"Total parameters: {total_stats['total_params']:,}")
    print(f"Total size: {format_size(total_stats['total_size_bytes'])}")
    print(f"Total size: {total_stats['total_size_bytes'] / (1024 * 1024):.2f} MB")

def get_dtype_string(dtype):
    """将ONNX数据类型转换为字符串表示"""
    dtype_mapping = {
        onnx.TensorProto.FLOAT: 'float32',
        onnx.TensorProto.UINT8: 'uint8',
        onnx.TensorProto.INT8: 'int8',
        onnx.TensorProto.UINT16: 'uint16',
        onnx.TensorProto.INT16: 'int16',
        onnx.TensorProto.INT32: 'int32',
        onnx.TensorProto.INT64: 'int64',
        onnx.TensorProto.STRING: 'string',
        onnx.TensorProto.BOOL: 'bool',
        onnx.TensorProto.FLOAT16: 'float16',
        onnx.TensorProto.DOUBLE: 'float64',
        onnx.TensorProto.UINT32: 'uint32',
        onnx.TensorProto.UINT64: 'uint64',
        onnx.TensorProto.COMPLEX64: 'complex64',
        onnx.TensorProto.COMPLEX128: 'complex128',
        onnx.TensorProto.BFLOAT16: 'bfloat16',
    }
    return dtype_mapping.get(dtype, f'unknown({dtype})')

def get_dtype_size(dtype):
    dtype_sizes = {1: 4, 10: 2, 11: 8, 6: 4, 7: 8}
    return dtype_sizes.get(dtype, 4)

def format_size(size_bytes):
    units = ['B', 'KB', 'MB', 'GB']
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python onnx_advanced.py <model.onnx>")
        sys.exit(1)
    
    advanced_onnx_analysis(sys.argv[1])