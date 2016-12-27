/*! FILE Main.cpp

    This file is a part of the class Vector4D.
     
    Author: Dmitry Anisimov
    Date: 22.06.2013
    Mail: danston@ymail.com
    Web: http://www.anisimovdmitry.com
 
*/

#include "Vector4D.h"

//! Here are some examples on how to use the class Vector4D.
int main()
{
    // To start using the class just include the file Vector4D.h in your project
    // and declare it as #include "Vector4D.h". Then include _Vector4D namespace directive.
    
    using namespace _Vector4D;
    
    // First, check the correctness of the class Vector4D for a float and double precision.
    // You can also perform this test for the int-like types, but some of the tests will fail due to their special construction.
    
    cout << endl << "Test for a float precision." << endl;
    Vector4DTest::runf<float>();
    cout << endl << "Test for a double precision." << endl;
    Vector4DTest::runf<double>();
    
    // If all the tests above gave a SUCCESS result, you can start working with the class.
    // More information about the class's functions can be found in the Vector4DTest::runf() function or directly in the class Vector4D itself.
    
    // For simplicity you can use some of the predefined types of the Vector4D.
    
    cout << endl;
    cout << "Using predefined types: " << endl << endl;
    
    Vector4Df vf(1.0001, 2.0001, 3.0001, 4.0001); // Vector4D with a float precision
    cout << "Float vf = " << vf << ";" << endl;
    
    cout.precision(10);
    Vector4Dd vd(1.00010001, 2.00010001, 3.00010001, 4.00010001); // Vector4D with a double precision
    cout << "Double vd = " << vd << ";" << endl;
    
    // Be careful with an int precision since some of the mathematical functions in the Vector4D will compute the result in the double precision and then round it to int!
    // After this rounding the final result can be different from what you expect.
    // E.g. if you compute the length of a vector (0, 0, 3, 4), then the result is 5 which is also an int and correct,
    // but if it is a vector (0, 0, 1, 1), then the result is sqrt(2) ~ 1.4 and after rounding it becomes 1 which is not what you expect.
    // If you are not sure that all your int numbers will give the int result after using some of the Vector4D functions,
    // it is better to switch to float or double type.
    
    Vector4Di vi(1, 2, 3, 4); // Vector4D with an int precision
    cout << "Int vi = " << vi << ";" << endl << endl;
    
    // Or you can use your own type.
    cout << "Using user-defined types, e.g. long double: " << endl << endl;
    
    // You can also use the function runf() to check the correctness of the class Vector4D for a long double precision, but due to the special construction of the tests
    // in the Vector4DTest class, some of them will fail!
    cout.precision(14);
    typedef long double ldouble;
    Vector4D<ldouble> v1; // Vector4D with long double precision, v1 = (0.0, 0.0, 0.0, 0.0) here.
    
    vector<ldouble> vec(4);
    vec[0] = 1.000100010001; vec[1] = 2.000100010001; vec[2] = 3.000100010001; vec[3] = 4.000100010001;
    
    v1 = vec; // v1 = (1.000100010001, 2.000100010001, 3.000100010001, 4.000100010001) here
    cout << "Long double v1 = " << v1 << ";" << endl;
    cout << "Length of v1 = " << v1.length() << ";" << endl << endl;
    
    v1 = Vector4D<ldouble>(3.000100010001); // v1 = (3.000100010001, 3.000100010001, 3.000100010001, 3.000100010001) here
    cout << "Long double v1 = " << v1 << ";" << endl;
    cout << "Normalized v1 = " << v1.normalized() << ";" << endl << endl;
    
    Vector4D<ldouble> v2 = 2.000100010001; // v2 = (2.000100010001, 2.000100010001, 2.000100010001, 2.000100010001) here
    v2 = v1 + Vector4D<ldouble>(2.000100010001, 1.000100010001, 0.000100010001, -1.000100010001); v2 += v1;
    cout << "Long double v2 = " << v2 << ";" << endl;
    cout << "Unsigned angle in radians between v1 and v2 = " << v1.unsignedAngleRad(v2) << ";" << endl;
    cout << "Scalar product between v1 and v2 = " << v1.scalarProduct(v2) << ";" << endl << endl;
    
    // All the available options you can find in the class Vector4D or Vector4DTest.
}
