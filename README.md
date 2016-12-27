# 2D/3D/4D Vectors Ver. 1.0.0

### Description

A set of stand-alone classes for handling vector-based operations. This package includes 2D, 3D, and 4D vector class implementations. 
You can also use the VectorXD header that includes all the above classes together. All the classes are templated and include the predefined 
types for the standard int, float, and double precisions.

##### NOTE: This code has been tested only on Mac OS!

### Run the code

In order to run the code, open terminal and type the following:

```bash
cd path_to_the_folder/vectors/Vector(_Choose_Version_)D
```
```bash
make
```
```bash
./main
```

### 2D Example

```C++
cout.precision(14);
typedef long double ldouble;

Vector2D<ldouble> v1;
    
vector<ldouble> vec(2);
vec[0] = 1.000100010001; vec[1] = 2.000100010001;
    
v1 = vec;

cout << "Long double v1 = " << v1 << ";" << endl;
cout << "Length of v1 = " << v1.length() << ";" << endl << endl;
    
v1 = Vector2D<ldouble>(3.000100010001);

cout << "Long double v1 = " << v1 << ";" << endl;
cout << "Normalized v1 = " << v1.normalized() << ";" << endl << endl;
    
Vector2D<ldouble> v2 = 2.000100010001;
v2 = v1 + Vector2D<ldouble>(1.000100010001, 0.000100010001); v2 += v1;

cout << "Long double v2 = " << v2 << ";" << endl;
cout << "Signed angle in radians between v1 and v2 = " << v1.signedAngleRad(v2) << ";" << endl;
cout << "Cross product between v1 and v2 = " << v1.crossProduct(v2) << ";" << endl << endl;
```

##### NOTE: For more examples see main.cpp for each vector class!

### Bugs

If you find any bugs, please report them to me, and I will try to fix them as soon as possible!
