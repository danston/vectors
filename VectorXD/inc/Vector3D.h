/*! FILE Vector3D.h

    Class Vector3D.

    The original code is due to:
    Author: Frank Firsching
    Date: 17.03.2001

    Reimplementation is due to:
    Author: Dmitry Anisimov
    Date: 22.06.2013
    Mail: danston@ymail.com
    Web: http://www.anisimovdmitry.com

*/

#ifndef Vector3D_H
#define Vector3D_H

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace std;

// _Vector3D namespace start

namespace _Vector3D {

template<typename T> class Vector3D;

//! Some predefined types for Vector3D ----->

//! Vector3D with int
typedef Vector3D<int> Vector3Di;

//! Vector3D with float
typedef Vector3D<float> Vector3Df;

//! Vector3D with double
typedef Vector3D<double> Vector3Dd;

//! Compute the Pi value up to T precision
template<typename T> inline T Pi()
{
    Vector3D<T>vStart = Vector3D<T>(1.0, 0.0, 0.0);
    Vector3D<T>vEnd = Vector3D<T>(-1.0, 0.0, 0.0);
    return vStart.unsignedAngleRad(vEnd);
}

//! This class provides functions to handle 3-dimensional vectors
template<typename T> class Vector3D
{
public:
    //! x coordinate of the vector
    T x;
    //! y coordinate of the vector
    T y;
    //! z coordinate of the vector
    T z;

    //! Default constructor
    Vector3D() { x = 0.0; y = 0.0; z = 0.0; }
    //! Initialize Vector3D with one explicit coordinate
    Vector3D(const T _x) { x = _x; y = _x; z = _x; }
    //! Initialize Vector3D with two explicit coordinates
    Vector3D(const T _x, const T _y) { x = _x; y = _y; z = _y; }
    //! Initialize Vector3D with three explicit coordinates
    Vector3D(const T _x, const T _y, const T _z) { x = _x; y = _y; z = _z; }
    //! Copy-constructor
    Vector3D(const Vector3D &v) { x = v.x; y = v.y; z = v.z; }
    //! Initialize Vector3D with a pointer to an array
    Vector3D(T* a) { x = a[0]; y = a[1]; z = a[2]; }
    //! Initialize Vector3D with a standard C++ vector
    Vector3D(const vector<T> &v)
    {
        const int size = v.size();
        if(size == 0) { x = 0.0; y = 0.0; z = 0.0; }
        if(size == 1) { x = v[0]; y = v[0]; z = v[0]; }
        if(size == 2) { x = v[0]; y = v[1]; z = v[1]; }
        if(size >= 3) { x = v[0]; y = v[1]; z = v[2]; }
    }

    //! Destructor
    ~Vector3D() { }

    //! Access to vector-coordinates with []-operator for a constant object
    inline T operator[](const int i) const;
    //! Access to vector-coordinates with []-operator for a manipulation
    inline T& operator[](const int i);

    //! Assign a value to a Vector3D
    inline Vector3D& operator=(const T scalar);
    //! Assign the values of a Vector3D to another one
    inline Vector3D& operator=(const Vector3D &v);
    //! Assign the values of a standard C++ array to a Vector3D
    inline Vector3D& operator=(T* a);
    //! Assign the values of a standard C++ vector to a Vector3D
    inline Vector3D& operator=(const vector<T> &v);

    //! Check for equality
    inline bool operator==(const Vector3D &v) const;
    //! Check for inequality
    inline bool operator!=(const Vector3D &v) const;

    //! Addition of 2 vectors
    inline Vector3D operator+(const Vector3D &v) const;
    //! Subtraction of 2 vectors
    inline Vector3D operator-(const Vector3D &v) const;
    //! Negate the vector
    inline Vector3D operator-() const;
    //! Add vector v to the operand
    inline void operator+=(const Vector3D &v);
    //! Subtract vector v from the operand
    inline void operator-=(const Vector3D &v);

    //! Multiply a vector by a scalar
    inline Vector3D operator*(const T scalar) const;
    //! Multiply the operand by a scalar
    inline void operator*=(const T scalar);
    //! Divide a vector by a scalar
    inline Vector3D operator/(const T scalar) const;
    //! Divide the operand by a scalar
    inline void operator/=(const T scalar);

    //! Calculate the scalar-product of two vectors
    inline T operator*(const Vector3D &v) const;
    //! Calculate the scalar-product of two vectors
    inline T scalarProduct(const Vector3D &v) const;

    //! Calculate the cross-product of two vectors
    inline Vector3D crossProduct(const Vector3D &v) const;
    //! Calculate the cross-product of two vectors
    inline Vector3D operator%(const Vector3D &v) const;
    //! Calculate the cross-product of two vectors and assign it immediately to the operand
    inline void operator%=(const Vector3D &v);

    //! Return the angle between the current vector and v in radians
    inline T unsignedAngleRad(const Vector3D &v) const;
    //! Return the angle between the current vector and v in degrees
    inline T angleDeg(const Vector3D &v) const;

    //! Return the squared length of the vector
    inline T squaredLength() const;
    //! Return the length of the vector
    inline T length() const;
    //! Return the L1-Norm of the vector (Taxicab norm or Manhattan norm)
    inline T L1Norm() const;
    //! Return the L2-Norm of the vector (Euclidean norm)
    inline T L2Norm() const;
    //! Return the LInfinity-Norm of the vector (Maximum norm, Uniform norm, or Supremum norm)
    inline T LInfNorm() const;

    //! Return the index of the smallest coordinate of the vector
    inline int indexOfTheSmallestCoordinate() const;
    //! Return the value of the smallest coordinate of the vector
    inline T theSmallestCoordinate() const;
    //! Return the index of the biggest coordinate of the vector
    inline int indexOfTheBiggestCoordinate() const;
    //! Return the value of the biggest coordinate of the vector
    inline T theBiggestCoordinate() const;

    //! Set the length of the vector to one, but the orientation stays the same
    inline void normalize();
    //! Return the normalized vector as a copy
    inline Vector3D normalized() const;

    //! Reflect the current vector accross a given v through the origin
    inline void reflect(const Vector3D &v);
    //! Return a vector reflected accross a given v through the origin
    Vector3D reflected(const Vector3D &v) const;

    //! Set all the coordinates of the vector to their absolute values
    inline void abs();
};

/// Operator [] ---------->

template<typename T> inline T Vector3D<T>::operator[](const int i) const
{
    assert( ( i >= 0 ) && ( i < 3 ) );
    return (&x)[i];
}

template<typename T> inline T& Vector3D<T>::operator[](const int i)
{
    assert( ( i >= 0 ) && ( i < 3 ));
    return (&x)[i];
}

/// Operator = ---------->

template<typename T> inline Vector3D<T>& Vector3D<T>::operator=(const T scalar)
{
    x = y = z = scalar;
    return *this;
}

template<typename T> inline Vector3D<T>& Vector3D<T>::operator=(const Vector3D &v)
{
    x = v.x; y = v.y; z = v.z;
    return *this;
}

template<typename T> inline Vector3D<T>& Vector3D<T>::operator=(T* a)
{
    x = a[0]; y = a[1]; z = a[2];
    return *this;
}

template<typename T> inline Vector3D<T>& Vector3D<T>::operator=(const vector<T> &v)
{
    const int size = v.size();
    if(size == 0) { x = 0.0;  y = 0.0;  z = 0.0;  }
    if(size == 1) { x = v[0]; y = v[0]; z = v[0]; }
    if(size == 2) { x = v[0]; y = v[1]; z = v[1]; }
    if(size >= 3) { x = v[0]; y = v[1]; z = v[2]; }
    return *this;
}

/// Operators == and != ---------->

template<typename T> inline bool Vector3D<T>::operator==(const Vector3D &v) const
{
    return ( fabs(x - v.x) <= numeric_limits<T>::epsilon() && fabs(y - v.y) <= numeric_limits<T>::epsilon() && fabs(z - v.z) <= numeric_limits<T>::epsilon() );
}

template<typename T> inline bool Vector3D<T>::operator!=(const Vector3D &v) const
{
    return ( fabs(x - v.x) > numeric_limits<T>::epsilon() || fabs(y - v.y) > numeric_limits<T>::epsilon() || fabs(z - v.z) > numeric_limits<T>::epsilon() );
}

/// Operators - and + ---------->

template<typename T> inline Vector3D<T> Vector3D<T>::operator+(const Vector3D &v) const
{
    return Vector3D<T>(x + v.x, y + v.y, z + v.z);
}

template<typename T> inline Vector3D<T> Vector3D<T>::operator-(const Vector3D &v) const
{
    return Vector3D<T>(x - v.x, y - v.y, z - v.z);
}

template<typename T> inline Vector3D<T> Vector3D<T>::operator-() const
{
    return Vector3D<T>(-x, -y, -z);
}

template<typename T> inline void Vector3D<T>::operator+=(const Vector3D &v)
{
    x += v.x; y += v.y; z += v.z;
}

template<typename T> inline void Vector3D<T>::operator-=(const Vector3D &v)
{
    x -= v.x; y -= v.y; z -= v.z;
}

/// Operators / and * ---------->

template<typename T> inline Vector3D<T> Vector3D<T>::operator*(const T scalar) const
{
    return Vector3D<T>(scalar*x, scalar*y, scalar*z);
}

template<typename T, typename S> inline Vector3D<T> operator*(const S scalar, const Vector3D<T> &v)
{
    return v*scalar;
}

template<typename T> inline void Vector3D<T>::operator*=(const T scalar)
{
    x *= scalar; y *= scalar; z *= scalar;
}

template<typename T> inline Vector3D<T> Vector3D<T>::operator/(const T scalar) const
{
    assert( fabs(scalar) > numeric_limits<T>::epsilon() );
    return Vector3D<T>(x / scalar, y / scalar, z / scalar);
}

template<typename T> inline void Vector3D<T>::operator/=(const T scalar)
{
    assert( fabs(scalar) > numeric_limits<T>::epsilon() );
    x /= scalar; y /= scalar; z /= scalar;
}

/// Scalar product ---------->

template<typename T> inline T Vector3D<T>::scalarProduct(const Vector3D &v) const
{
    return (x*v.x + y*v.y + z*v.z);
}

template<typename T> inline T Vector3D<T>::operator*(const Vector3D &v) const
{
    return this->scalarProduct(v);
}

/// Cross product ---------->

template<typename T> inline Vector3D<T> Vector3D<T>::crossProduct(const Vector3D &v) const
{
    return Vector3D<T>(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
}

template<typename T> inline Vector3D<T> Vector3D<T>::operator%(const Vector3D &v) const
{
    return this->crossProduct(v);
}

template<typename T> inline void Vector3D<T>::operator%=(const Vector3D &v)
{
    (*this) = this->crossProduct(v);
}

/// Angles ---------->

template<typename T> inline T Vector3D<T>::unsignedAngleRad(const Vector3D &v) const
{
    return acos(this->scalarProduct(v) / (this->length()*v.length()));
}

template<typename T> inline T Vector3D<T>::angleDeg(const Vector3D &v) const
{
    return this->unsignedAngleRad(v)*(180.0 / Pi<T>());
}

/// Norms ---------->

template<typename T> inline T Vector3D<T>::squaredLength() const
{
    return (x*x + y*y + z*z);
}

template<typename T> inline T Vector3D<T>::length() const
{
    return sqrt(this->squaredLength());
}

template<typename T> inline T Vector3D<T>::L1Norm() const
{
    return fabs(x) + fabs(y) + fabs(z);
}

template<typename T> inline T Vector3D<T>::L2Norm() const
{
    return this->length();
}

template<typename T> inline T Vector3D<T>::LInfNorm() const
{
    const T max_x_y = max(fabs(x), fabs(y));
    return max(max_x_y, fabs(z));
}

/// The smallest and the biggest coordinates ---------->

template<typename T> inline int Vector3D<T>::indexOfTheSmallestCoordinate() const
{
    return ( (x < y) ? ((x < z) ? 0 : 2) : ((y < z) ? 1 : 2) );
}

template<typename T> inline T Vector3D<T>::theSmallestCoordinate() const
{
    return (&x)[this->indexOfTheSmallestCoordinate()];
}

template<typename T> inline int Vector3D<T>::indexOfTheBiggestCoordinate() const
{
    return ( (x < y) ? ((y < z) ? 2 : 1) : ((x < z) ? 2 : 0) );
}

template<typename T> inline T Vector3D<T>::theBiggestCoordinate() const
{
    return (&x)[this->indexOfTheBiggestCoordinate()];
}

/// Normalization ---------->

template<typename T> inline void Vector3D<T>::normalize()
{
    (*this) = this->normalized();
}

template<typename T> inline Vector3D<T> Vector3D<T>::normalized() const
{
    T thisLength = this->length();
    if(fabs(thisLength) >  numeric_limits<T>::epsilon()) return Vector3D<T>(x / thisLength, y / thisLength, z / thisLength);
    else return Vector3D<T>(0.0, 0.0, 1.0);
}

/// Reflection ---------->

template<typename T> inline void Vector3D<T>::reflect(const Vector3D &v)
{
    (*this) = this->reflected(v);
}

template<typename T> inline Vector3D<T> Vector3D<T>::reflected(const Vector3D &v) const
{
    return (2.0*(((*this)*v) / (v*v))*v - (*this));
}

/// Absolute value ---------->

template<typename T> inline void Vector3D<T>::abs()
{
    x = fabs(x); y = fabs(y); z = fabs(z);
}

/// Read from or write to stream ---------->

//! Write a 3D-vector to a stream
template<typename T> inline ostream& operator<<(ostream &ostr, const Vector3D<T> &v)
{
    return (ostr << "(" << v.x << ", " << v.y << ", " << v.z << ")");
}

//! Read a 3D-vector from a stream
template<typename T> inline istream& operator>>(istream &istr, Vector3D<T> &v)
{
    return (istr >> v.x >> v.y >> v.z);
}

}

// _Vector3D namespace end

//! This class provides the function to check the correctness of the class Vector3D
class Vector3DTest
{
public:
    Vector3DTest() { }
    ~Vector3DTest() { }

    //! A static function which tests Vector3D class for correctness
    template<class T> static void runf()
    {
        const int numberOfTests = 47;
        int passed = 0;

        cout << endl << "!Test started....." << endl << endl;

        /// Constructors ---------->

        _Vector3D::Vector3D<T> v_1;

        if( fabs(v_1.x) <= numeric_limits<T>::epsilon() && fabs(v_1.y) <= numeric_limits<T>::epsilon() && fabs(v_1.z) <= numeric_limits<T>::epsilon() ) {
            cout << "Default constructor test: PASSED." << endl;
            passed++;
        } else cout << "Default constructor test: FAILED." << endl;

        _Vector3D::Vector3D<T> v_2_1(1.0);

        if( fabs(v_2_1.x - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_2_1.y - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_2_1.z - 1.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constructor with one explicit coordinate test 1: PASSED." << endl;
            passed++;
        } else cout << "Constructor with one explicit coordinate test 1: FAILED." << endl;

        T scalar = 3.0;
        _Vector3D::Vector3D<T> v_2_2 = scalar;

        if( fabs(v_2_2.x - 3.0) <= numeric_limits<T>::epsilon() && fabs(v_2_2.y - 3.0) <= numeric_limits<T>::epsilon() && fabs(v_2_2.z - 3.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constructor with one explicit coordinate test 2: PASSED." << endl;
            passed++;
        } else cout << "Constructor with one explicit coordinate test 2: FAILED." << endl;

        _Vector3D::Vector3D<T> v_3(v_2_1.x, 2.0);

        if( fabs(v_3.x - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_3.y - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_3.z - 2.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constructor with two explicit coordinates test: PASSED." << endl;
            passed++;
        } else cout << "Constructor with two explicit coordinates test: FAILED." << endl;

        _Vector3D::Vector3D<T> v_4(v_2_1.x, 2.0, 1.0);

        if( fabs(v_4.x - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_4.y - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_4.z - 1.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constructor with three explicit coordinates test: PASSED." << endl;
            passed++;
        } else cout << "Constructor with three explicit coordinates test: FAILED." << endl;

        _Vector3D::Vector3D<T> v_4_1(v_3);

        if( fabs(v_4_1.x - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_4_1.y - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_4_1.z - 2.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Copy constructor test 1: PASSED." << endl;
            passed++;
        } else cout << "Copy constructor test 1: FAILED." << endl;

        _Vector3D::Vector3D<T> v_4_2 = v_4_1;

        if( fabs(v_4_2.x - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_4_2.y - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_4_2.z - 2.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Copy constructor test 2: PASSED." << endl;
            passed++;
        } else cout << "Copy constructor test 2: FAILED." << endl;

        T a[3] = { 2.0, 2.0, 2.0 };
        _Vector3D::Vector3D<T> v_5_1(a);

        if( fabs(v_5_1.x - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_5_1.y - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_5_1.z - 2.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constructor from an array test 1: PASSED." << endl;
            passed++;
        } else cout << "Constructor from an array test 1: FAILED." << endl;

        _Vector3D::Vector3D<T> v_5_2 = a;

        if( fabs(v_5_2.x - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_5_2.y - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_5_2.z - 2.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constructor from an array test 2: PASSED." << endl;
            passed++;
        } else cout << "Constructor from an array test 2: FAILED." << endl;

        vector<T> v(3);
        v[0] = 1.0; v[1] = 2.0; v[2] = 1.0;
        _Vector3D::Vector3D<T> v_6_1(v);

        if( fabs(v_6_1[0] - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_6_1[1] - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_6_1[2] - 1.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constructor from a standard C++ vector test 1: PASSED." << endl;
            passed++;
        } else cout << "Constructor from a standard C++ vector test 1: FAILED." << endl;

        _Vector3D::Vector3D<T> v_6_2 = v;

        if( fabs(v_6_2[0] - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_6_2[1] - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_6_2[2] - 1.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constructor from a standard C++ vector test 2: PASSED." << endl;
            passed++;
        } else cout << "Constructor from a standard C++ vector test 2: FAILED." << endl;

        /// Operator [] ---------->

        const _Vector3D::Vector3D<T> v_7(1.0, 2.0, 3.0);

        if( fabs(v_7[0] - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_7[1] - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_7[2] - 3.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constant object []-operator test: PASSED." << endl;
            passed++;
        } else cout << "Constant object []-operator test: FAILED." << endl;

        _Vector3D::Vector3D<T> v_8(1.0, 2.0, 3.0);
        T sum = 0.0;
        for(int i = 0; i < 3; ++i) sum += v_8[i];

        if( fabs(sum - 6.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Manipulation []-operator test: PASSED." << endl;
            passed++;
        } else cout << "Manipulation []-operator test: FAILED." << endl;

        /// Operator = ---------->

        _Vector3D::Vector3D<T> v_9;
        v_9 = scalar;

        if( fabs(v_9.x - 3.0) <= numeric_limits<T>::epsilon() && fabs(v_9.y - 3.0) <= numeric_limits<T>::epsilon() && fabs(v_9.z - 3.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Operator=(scalar) test: PASSED." << endl;
            passed++;
        } else cout << "Operator=(scalar) test: FAILED." << endl;

        _Vector3D::Vector3D<T> v_10;
        v_10 = v_9;

        if( fabs(v_10.x - 3.0) <= numeric_limits<T>::epsilon() && fabs(v_10.y - 3.0) <= numeric_limits<T>::epsilon() && fabs(v_10.z - 3.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Operator=(Vector3D) test: PASSED." << endl;
            passed++;
        } else cout << "Operator=(Vector3D) test: FAILED." << endl;

        _Vector3D::Vector3D<T> v_11;
        v_11 = a;

        if( fabs(v_11.x - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_11.y - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_11.z - 2.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Operator=(C++ array) test: PASSED." << endl;
            passed++;
        } else cout << "Operator=(C++ array) test: FAILED." << endl;

        _Vector3D::Vector3D<T> v_12;
        v_12 = v;

        if( fabs(v_12.x - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_12.y - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_12.z - 1.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Operator=(C++ vector) test: PASSED." << endl;
            passed++;
        } else cout << "Operator=(C++ vector) test: FAILED." << endl;

        /// Operators == and != ---------->

        _Vector3D::Vector3D<T> v_13_1(1.0, 2.0, 3.0);
        _Vector3D::Vector3D<T> v_13_2(1.0, 3.0, 2.0);

        if((v_13_1 == v_13_2) == false) {
            cout << "Operator== test 1: PASSED." << endl;
            passed++;
        } else cout << "Operator== test 1: FAILED." << endl;

        if((v_13_1 == v_13_1) == true) {
            cout << "Operator== test 2: PASSED." << endl;
            passed++;
        } else cout << "Operator== test 2: FAILED." << endl;

        if((v_13_1 != v_13_2) == true) {
            cout << "Operator!= test 1: PASSED." << endl;
            passed++;
        } else cout << "Operator!= test 1: FAILED." << endl;

        if((v_13_2 != v_13_2) == false) {
            cout << "Operator!= test 2: PASSED." << endl;
            passed++;
        } else cout << "Operator!= test 2: FAILED." << endl;

        /// Operators - and + ---------->

        _Vector3D::Vector3D<T> v_14(1.0, 1.0, 1.0);
        _Vector3D::Vector3D<T> v_15(2.0, 2.0, 2.0);
        _Vector3D::Vector3D<T> v_res(3.0, 3.0, 3.0);

        if( (v_14 + v_15) ==  v_res ) {
            cout << "Operator + test: PASSED." << endl;
            passed++;
        } else cout << "Operator + test: FAILED." << endl;

        v_res = _Vector3D::Vector3D<T>(-1.0, -1.0, -1.0);

        if( (v_14 - v_15) ==  v_res ) {
            cout << "Operator - (Subtraction) test: PASSED." << endl;
            passed++;
        } else cout << "Operator - (Subtraction) test: FAILED." << endl;

        if( -v_14 ==  v_res ) {
            cout << "Operator - (Negate) test: PASSED." << endl;
            passed++;
        } else cout << "Operator - (Negate) test: FAILED." << endl;

        v_14 -= v_15;
        if( v_14 ==  v_res ) {
            cout << "Operator -= test: PASSED." << endl;
            passed++;
        } else cout << "Operator -= test: FAILED." << endl;

        v_res = _Vector3D::Vector3D<T>(3.0, 3.0, 3.0);

        v_14 = _Vector3D::Vector3D<T>(1.0, 1.0, 1.0);
        v_14 += v_15;
        if( v_14 ==  v_res ) {
            cout << "Operator += test: PASSED." << endl;
            passed++;
        } else cout << "Operator += test: FAILED." << endl;

        /// Operators / and * ---------->

        v_14 = _Vector3D::Vector3D<T>(1.0, 1.0, 1.0);

        scalar = 2.0;

        if( (v_14*scalar) ==  v_15 ) {
            cout << "Operator*(Scalar) from the right test: PASSED." << endl;
            passed++;
        } else cout << "Operator*(Scalar) from the right test: FAILED." << endl;

        if( (scalar*v_14) ==  v_15 ) {
            cout << "Operator*(Scalar) from the left test: PASSED." << endl;
            passed++;
        } else cout << "Operator*(Scalar) from the left test: FAILED." << endl;

        v_14 *= scalar;
        if( v_14 ==  v_15 ) {
            cout << "Operator *= test: PASSED." << endl;
            passed++;
        } else cout << "Operator *= test: FAILED." << endl;

        v_14 = _Vector3D::Vector3D<T>(1.0, 1.0, 1.0);
        v_res = _Vector3D::Vector3D<T>(0.5, 0.5, 0.5);

        if( (v_14 / scalar) ==  v_res ) {
            cout << "Operator / test: PASSED." << endl;
            passed++;
        } else cout << "Operator / test: FAILED." << endl;

        v_14 /= scalar;
        if( v_14 ==  v_res ) {
            cout << "Operator /= test: PASSED." << endl;
            passed++;
        } else cout << "Operator /= test: FAILED." << endl;

        /// Scalar product ---------->

        v_14 = _Vector3D::Vector3D<T>(1.0, 1.0, 1.0);

        if( fabs(v_14*v_15 - 6.0) <= numeric_limits<T>::epsilon() && fabs(v_14.scalarProduct(v_15) - 6.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Scalar product test: PASSED." << endl;
            passed++;
        } else cout << "Scalar product test: FAILED." << endl;

        /// Cross product ---------->

        if( (v_14%v_15) == _Vector3D::Vector3D<T>() && (v_14.crossProduct(v_15)) == _Vector3D::Vector3D<T>() ) {
            cout << "Cross product test 1: PASSED." << endl;
            passed++;
        } else cout << "Cross product test 1: FAILED." << endl;

        v_14 %= v_15;
        if( v_14 == _Vector3D::Vector3D<T>() ) {
            cout << "Cross product test 2: PASSED." << endl;
            passed++;
        } else cout << "Cross product test 2: FAILED." << endl;

        /// Norms ---------->

        v_15 = _Vector3D::Vector3D<T>(0.0, 3.0, 4.0);

        if( fabs(v_15.squaredLength() - 25.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Squared length test: PASSED." << endl;
            passed++;
        } else cout << "Squared length test: FAILED." << endl;

        if( fabs(v_15.L1Norm() - 7.0) <= numeric_limits<T>::epsilon() ) {
            cout << "L1 norm test: PASSED." << endl;
            passed++;
        } else cout << "L1 norm test: FAILED." << endl;

        if( fabs(v_15.length() - 5.0) <= numeric_limits<T>::epsilon() && fabs(v_15.L2Norm() - 5.0) <= numeric_limits<T>::epsilon() ) {
            cout << "L2 norm/length test: PASSED." << endl;
            passed++;
        } else cout << "L2 norm/length test: FAILED." << endl;

        if( fabs(v_15.LInfNorm() - 4.0) <= numeric_limits<T>::epsilon() ) {
            cout << "LInfinity norm test: PASSED." << endl;
            passed++;
        } else cout << "LInfinity norm test: FAILED." << endl;

        if( v_15.indexOfTheSmallestCoordinate() == 0) {
            cout << "Index of the smallest coordinate test: PASSED." << endl;
            passed++;
        } else cout << "Index of the smallest coordinate test: FAILED." << endl;

        /// The smallest and the biggest coordinates ---------->

        if( fabs(v_15.theSmallestCoordinate()) <= numeric_limits<T>::epsilon() ) {
            cout << "The smallest coordinate test: PASSED." << endl;
            passed++;
        } else cout << "The smallest coordinate test: FAILED." << endl;

        if( v_15.indexOfTheBiggestCoordinate() == 2) {
            cout << "Index of the biggest coordinate test: PASSED." << endl;
            passed++;
        } else cout << "Index of the biggest coordinate test: FAILED." << endl;

        if( fabs(v_15.theBiggestCoordinate() - 4.0) <= numeric_limits<T>::epsilon() ) {
            cout << "The biggest coordinate test: PASSED." << endl;
            passed++;
        } else cout << "The biggest coordinate test: FAILED." << endl;

        /// Normalization ---------->

        v_15.normalize();
        if( fabs(v_15.x) <= numeric_limits<T>::epsilon() && fabs(v_15.y - 0.6) <= numeric_limits<T>::epsilon() && fabs(v_15.z - 0.8) <= numeric_limits<T>::epsilon() ) {
            cout << "Normalize test: PASSED." << endl;
            passed++;
        } else cout << "Normalize test: FAILED." << endl;

        if( (_Vector3D::Vector3D<T>(0.0, 0.0, 0.0).normalized() == _Vector3D::Vector3D<T>(0.0, 0.0, 1.0)) == true) {
            cout << "Normalized test: PASSED." << endl;
            passed++;
        } else cout << "Normalized test: FAILED." << endl;

        /// Reflection ---------->

        v_14 = _Vector3D::Vector3D<T>(-1.0, 0.0, 0.0);
        v_15 = _Vector3D::Vector3D<T>(0.0, 0.0, 1.0);

        v_14.reflect(v_15);
        if( fabs(v_14.x - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_14.y) <= numeric_limits<T>::epsilon() && fabs(v_14.z) <= numeric_limits<T>::epsilon() ) {
            cout << "Reflect test: PASSED." << endl;
            passed++;
        } else cout << "Reflect test: FAILED." << endl;

        if( (_Vector3D::Vector3D<T>(-1.0, 0.0, 0.0).reflected(v_15) == _Vector3D::Vector3D<T>(1.0, 0.0, 0.0)) == true) {
            cout << "Reflected test: PASSED." << endl;
            passed++;
        } else cout << "Reflected test: FAILED." << endl;

        /// Absolute value ---------->

        v_14 = _Vector3D::Vector3D<T>(3.0, 4.0, 5.0);
        v_15 = _Vector3D::Vector3D<T>(-3.0, -4.0, -5.0);
        v_15.abs();
        if( fabs(v_15.x - v_14.x) <= numeric_limits<T>::epsilon() && fabs(v_15.y - v_14.y) <= numeric_limits<T>::epsilon() && fabs(v_15.z - v_14.z) <= numeric_limits<T>::epsilon() ) {
            cout << "Absolute value test: PASSED." << endl;
            passed++;
        } else cout << "Absolute value test: FAILED." << endl;

        cout << endl << ".....Test finished!" << endl;

        cout << endl;
        cout << "Overall test result: " << endl << endl;
        cout << "Number of tests = " << numberOfTests << ";" << endl;
        cout << "Tests passed: " << passed << "/" << numberOfTests << ";" << endl;
        cout << "Tests failed: " << numberOfTests - passed << "/" << numberOfTests << ";" << endl;

        if(numberOfTests == passed) cout << "TestVector3DClass: SUCCESS!" << endl;
        else cout << "TestVector3DClass: FAILED!" << endl;
    }
};

#endif // Vector3D_H
