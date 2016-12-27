/*! FILE Main.cpp

    This file is a part of the library VectorXD.
     
    Author: Dmitry Anisimov
    Date: 22.06.2013
    Mail: danston@ymail.com
    Web: http://www.anisimovdmitry.com
 
*/

#include <VectorXD>

//! Here are some examples on how to use the library VectorXD.
int main()
{
    // To start using the library just include the file VectorXD in your project
    // and declare it as #include <VectorXD>. Then include _VectorXD namespace directive.
    
    using namespace _VectorXD;
    
    // First, check the correctness of the classes Vector2D, Vector3D, and Vector4D for a double precision.
    cout << endl << "Test Vector2D: " << endl;
    Vector2DTest::runf<double>();    
    cout << endl << "Test Vector3D: " << endl;
    Vector3DTest::runf<double>();
    cout << endl << "Test Vector4D: " << endl;
    Vector4DTest::runf<double>();
    cout << endl;
    
    // Now construct a simple example with homogeneous coordinates
    // using Vector3D and Vector4D classes with a double precision.
    
    cout.precision(10);
    Vector3Dd v3d;
    cout << "v3d = " << v3d << ";" << endl << endl;
    
    Vector4Dd v4d(1.00010001, 2.00010001, 3.00010001, 4.00010001);
    cout << "v4d = " << v4d << ";" << endl << endl;
    
    v4d.cartesianize();
    v3d = Vector3Dd(v4d.x, v4d.y, v4d.z);
    
    cout << "v3d with cartesian coordinates corresponding to the homogeneous coordinates in v4d = " << v3d << ";" << endl << endl;
    
    cout << "v3d = " << v3d << ";" << endl << endl;
    
    v4d = Vector4Dd(v3d.x * 4.00010001, v3d.y * 4.00010001, v3d.z * 4.00010001, 4.00010001);
    
    cout << "Homogeneous coordinates in v4d for v3d = " << v4d << ";" << endl << endl;
}
