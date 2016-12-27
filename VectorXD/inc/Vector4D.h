/*! FILE Vector4D.h

    Class Vector4D.

    The original code is due to:
    Author: Frank Firsching
    Date: 17.03.2001

    Reimplementation is due to:
    Author: Dmitry Anisimov
    Date: 22.06.2013
    Mail: danston@ymail.com
    Web: http://www.anisimovdmitry.com

*/

#ifndef Vector4D_H
#define Vector4D_H

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace std;

// _Vector4D namespace start

namespace _Vector4D {

template<typename T> class Vector4D;

//! Some predefined types for Vector4D ----->

//! Vector4D with int
typedef Vector4D<int> Vector4Di;

//! Vector4D with float
typedef Vector4D<float> Vector4Df;

//! Vector4D with double
typedef Vector4D<double> Vector4Dd;

//! Compute the Pi value up to T precision
template<typename T> inline T Pi()
{
    Vector4D<T>vStart = Vector4D<T>(1.0, 0.0, 0.0, 0.0);
    Vector4D<T>vEnd = Vector4D<T>(-1.0, 0.0, 0.0, 0.0);
    return vStart.unsignedAngleRad(vEnd);
}

//! This class provides functions to handle 4-dimensional vectors
template<typename T> class Vector4D
{
public:
    //! x coordinate of the vector
    T x;
    //! y coordinate of the vector
    T y;
    //! z coordinate of the vector
    T z;
    //! w coordinate of the vector
    T w;

    //! Default constructor
    Vector4D() { x = 0.0; y = 0.0; z = 0.0; w = 0.0; }
    //! Initialize Vector4D with one explicit coordinate
    Vector4D(const T _x) { x = _x; y = _x; z = _x; w = _x; }
    //! Initialize Vector4D with two explicit coordinates
    Vector4D(const T _x, const T _y) { x = _x; y = _y; z = _y; w = _y; }
    //! Initialize Vector4D with three explicit coordinates
    Vector4D(const T _x, const T _y, const T _z) { x = _x; y = _y; z = _z; w = _z; }
    //! Initialize Vector4D with four explicit coordinates
    Vector4D(const T _x, const T _y, const T _z, const T _w) { x = _x; y = _y; z = _z; w = _w; }
    //! Copy-constructor
    Vector4D(const Vector4D &v) { x = v.x; y = v.y; z = v.z; w = v.w; }
    //! Initialize Vector4D with a pointer to an array
    Vector4D(T* a) { x = a[0]; y = a[1]; z = a[2]; w = a[3]; }
    //! Initialize Vector4D with a standard C++ vector
    Vector4D(const vector<T> &v)
    {
        const int size = v.size();
        if(size == 0) { x = 0.0; y = 0.0; z = 0.0; w = 0.0; }
        if(size == 1) { x = v[0]; y = v[0]; z = v[0]; w = v[0]; }
        if(size == 2) { x = v[0]; y = v[1]; z = v[1]; w = v[1]; }
        if(size == 3) { x = v[0]; y = v[1]; z = v[2]; w = v[2]; }
        if(size >= 4) { x = v[0]; y = v[1]; z = v[2]; w = v[3]; }
    }

    //! Destructor
    ~Vector4D() { }

    //! Access to vector-coordinates with []-operator for a constant object
    inline T operator[](const int i) const;
    //! Access to vector-coordinates with []-operator for a manipulation
    inline T& operator[](const int i);

    //! Assign a value to a Vector4D
    inline Vector4D& operator=(const T scalar);
    //! Assign the values of a Vector4D to another one
    inline Vector4D& operator=(const Vector4D &v);
    //! Assign the values of a standard C++ array to a Vector4D
    inline Vector4D& operator=(T* a);
    //! Assign the values of a standard C++ vector to a Vector4D
    inline Vector4D& operator=(const vector<T> &v);

    //! Check for equality
    inline bool operator==(const Vector4D &v) const;
    //! Check for inequality
    inline bool operator!=(const Vector4D &v) const;

    //! Addition of 2 vectors
    inline Vector4D operator+(const Vector4D &v) const;
    //! Subtraction of 2 vectors
    inline Vector4D operator-(const Vector4D &v) const;
    //! Negate the vector
    inline Vector4D operator-() const;
    //! Add vector v to the operand
    inline void operator+=(const Vector4D &v);
    //! Subtract vector v from the operand
    inline void operator-=(const Vector4D &v);

    //! Multiply a vector by a scalar
    inline Vector4D operator*(const T scalar) const;
    //! Multiply the operand by a scalar
    inline void operator*=(const T scalar);
    //! Divide a vector by a scalar
    inline Vector4D operator/(const T scalar) const;
    //! Divide the operand by a scalar
    inline void operator/=(const T scalar);

    //! Calculate the scalar-product of two vectors
    inline T operator*(const Vector4D &v) const;
    //! Calculate the scalar-product of two vectors
    inline T scalarProduct(const Vector4D &v) const;

    //! Return the angle between the current vector and v in radians
    inline T unsignedAngleRad(const Vector4D &v) const;
    //! Return the angle between the current vector and v in degrees
    inline T angleDeg(const Vector4D &v) const;

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

    //! Project a homogenous vector to 3D-space (results in w = 1)
    inline void cartesianize();
    //! Return the projection of a homogenous vector to 3D-space (results in w = 1)
    inline Vector4D cartesianized() const;

    //! Set the length of the vector to one, but the orientation stays the same
    inline void normalize();
    //! Return the normalized vector as a copy
    inline Vector4D normalized() const;

    //! Reflect the current vector accross a given v through the origin
    inline void reflect(const Vector4D &v);
    //! Return a vector reflected accross a given v through the origin
    Vector4D reflected(const Vector4D &v) const;

    //! Set all the coordinates of the vector to their absolute values
    inline void abs();
};

/// Operator [] ---------->

template<typename T> inline T Vector4D<T>::operator[](const int i) const
{
    assert( ( i >= 0 ) && ( i < 4 ) );
    return (&x)[i];
}

template<typename T> inline T& Vector4D<T>::operator[](const int i)
{
    assert( ( i >= 0 ) && ( i < 4 ));
    return (&x)[i];
}

/// Operator = ---------->

template<typename T> inline Vector4D<T>& Vector4D<T>::operator=(const T scalar)
{
    x = y = z = w = scalar;
    return *this;
}

template<typename T> inline Vector4D<T>& Vector4D<T>::operator=(const Vector4D &v)
{
    x = v.x; y = v.y; z = v.z; w = v.w;
    return *this;
}

template<typename T> inline Vector4D<T>& Vector4D<T>::operator=(T* a)
{
    x = a[0]; y = a[1]; z = a[2]; w = a[3];
    return *this;
}

template<typename T> inline Vector4D<T>& Vector4D<T>::operator=(const vector<T> &v)
{
    const int size = v.size();
    if(size == 0) { x = 0.0; y = 0.0; z = 0.0; w = 0.0; }
    if(size == 1) { x = v[0]; y = v[0]; z = v[0]; w = v[0]; }
    if(size == 2) { x = v[0]; y = v[1]; z = v[1]; w = v[1]; }
    if(size == 3) { x = v[0]; y = v[1]; z = v[2]; w = v[2]; }
    if(size >= 4) { x = v[0]; y = v[1]; z = v[2]; w = v[3]; }
    return *this;
}

/// Operators == and != ---------->

template<typename T> inline bool Vector4D<T>::operator==(const Vector4D &v) const
{
    return ( fabs(x - v.x) <= numeric_limits<T>::epsilon() && fabs(y - v.y) <= numeric_limits<T>::epsilon() && fabs(z - v.z) <= numeric_limits<T>::epsilon() && fabs(w - v.w) <= numeric_limits<T>::epsilon() );
}

template<typename T> inline bool Vector4D<T>::operator!=(const Vector4D &v) const
{
    return ( fabs(x - v.x) > numeric_limits<T>::epsilon() || fabs(y - v.y) > numeric_limits<T>::epsilon() || fabs(z - v.z) > numeric_limits<T>::epsilon() || fabs(w - v.w) > numeric_limits<T>::epsilon() );
}

/// Operators - and + ---------->

template<typename T> inline Vector4D<T> Vector4D<T>::operator+(const Vector4D &v) const
{
    return Vector4D<T>(x + v.x, y + v.y, z + v.z, w + v.w);
}

template<typename T> inline Vector4D<T> Vector4D<T>::operator-(const Vector4D &v) const
{
    return Vector4D<T>(x - v.x, y - v.y, z - v.z, w - v.w);
}

template<typename T> inline Vector4D<T> Vector4D<T>::operator-() const
{
    return Vector4D<T>(-x, -y, -z, -w);
}

template<typename T> inline void Vector4D<T>::operator+=(const Vector4D &v)
{
    x += v.x; y += v.y; z += v.z; w += v.w;
}

template<typename T> inline void Vector4D<T>::operator-=(const Vector4D &v)
{
    x -= v.x; y -= v.y; z -= v.z; w -= v.w;
}

/// Operators / and * ---------->

template<typename T> inline Vector4D<T> Vector4D<T>::operator*(const T scalar) const
{
    return Vector4D<T>(scalar*x, scalar*y, scalar*z, scalar*w);
}

template<typename T, typename S> inline Vector4D<T> operator*(const S scalar, const Vector4D<T> &v)
{
    return v*scalar;
}

template<typename T> inline void Vector4D<T>::operator*=(const T scalar)
{
    x *= scalar; y *= scalar; z *= scalar; w *= scalar;
}

template<typename T> inline Vector4D<T> Vector4D<T>::operator/(const T scalar) const
{
    assert( fabs(scalar) > numeric_limits<T>::epsilon() );
    return Vector4D<T>(x / scalar, y / scalar, z / scalar, w / scalar);
}

template<typename T> inline void Vector4D<T>::operator/=(const T scalar)
{
    assert( fabs(scalar) > numeric_limits<T>::epsilon() );
    x /= scalar; y /= scalar; z /= scalar; w /= scalar;
}

/// Scalar product ---------->

template<typename T> inline T Vector4D<T>::scalarProduct(const Vector4D &v) const
{
    return (x*v.x + y*v.y + z*v.z + w*v.w);
}

template<typename T> inline T Vector4D<T>::operator*(const Vector4D &v) const
{
    return this->scalarProduct(v);
}

/// Angles ---------->

template<typename T> inline T Vector4D<T>::unsignedAngleRad(const Vector4D &v) const
{
    return acos(this->scalarProduct(v) / (this->length()*v.length()));
}

template<typename T> inline T Vector4D<T>::angleDeg(const Vector4D &v) const
{
    return this->unsignedAngleRad(v)*(180.0 / Pi<T>());
}

/// Norms ---------->

template<typename T> inline T Vector4D<T>::squaredLength() const
{
    return (x*x + y*y + z*z + w*w);
}

template<typename T> inline T Vector4D<T>::length() const
{
    return sqrt(this->squaredLength());
}

template<typename T> inline T Vector4D<T>::L1Norm() const
{
    return fabs(x) + fabs(y) + fabs(z) + fabs(w);
}

template<typename T> inline T Vector4D<T>::L2Norm() const
{
    return this->length();
}

template<typename T> inline T Vector4D<T>::LInfNorm() const
{
    const T max_x_y = max(fabs(x), fabs(y));
    const T max_x_y_z = max(max_x_y, fabs(z));
    return max(max_x_y_z, fabs(w));
}

/// The smallest and the biggest coordinates ---------->

template<typename T> inline int Vector4D<T>::indexOfTheSmallestCoordinate() const
{
    return ( (x < y) ? ((x < z) ? ((x < w) ? 0 : 3) : ((z < w) ? 2 : 3)) : ((y < z) ? ((y < w) ? 1 : 3) : ((z < w) ? 2 : 3)) );
}

template<typename T> inline T Vector4D<T>::theSmallestCoordinate() const
{
    return (&x)[this->indexOfTheSmallestCoordinate()];
}

template<typename T> inline int Vector4D<T>::indexOfTheBiggestCoordinate() const
{
    return ( (x < y) ? ((y < z) ? ((z < w) ? 3 : 2) : ((y < w) ? 3 : 1)) : ((x < z) ? ((z < w) ? 3 : 2) : ((x < w) ? 3 : 0)) );
}

template<typename T> inline T Vector4D<T>::theBiggestCoordinate() const
{
    return (&x)[this->indexOfTheBiggestCoordinate()];
}

/// Cartesianization ---------->

template<typename T> inline void Vector4D<T>::cartesianize()
{
    (*this) = this->cartesianized();
}

template<typename T> inline Vector4D<T> Vector4D<T>::cartesianized() const
{
    return Vector4D<T>(x / w, y / w, z / w, 1.0);
}

/// Normalization ---------->

template<typename T> inline void Vector4D<T>::normalize()
{
    (*this) = this->normalized();
}

template<typename T> inline Vector4D<T> Vector4D<T>::normalized() const
{
    T thisLength = this->length();
    if(fabs(thisLength) >  numeric_limits<T>::epsilon()) return Vector4D<T>(x / thisLength, y / thisLength, z / thisLength, w / thisLength);
    else return Vector4D<T>(0.0, 0.0, 0.0, 1.0);
}

/// Reflection ---------->

template<typename T> inline void Vector4D<T>::reflect(const Vector4D &v)
{
    (*this) = this->reflected(v);
}

template<typename T> inline Vector4D<T> Vector4D<T>::reflected(const Vector4D &v) const
{
    return (2.0*(((*this)*v) / (v*v))*v - (*this));
}

/// Absolute value ---------->

template<typename T> inline void Vector4D<T>::abs()
{
    x = fabs(x); y = fabs(y); z = fabs(z); w = fabs(w);
}

/// Read from or write to stream ---------->

//! Write a 4D-vector to a stream
template<typename T> inline ostream& operator<<(ostream &ostr, const Vector4D<T> &v)
{
    return (ostr << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")");
}

//! Read a 4D-vector from a stream
template<typename T> inline istream& operator>>(istream &istr, Vector4D<T> &v)
{
    return (istr >> v.x >> v.y >> v.z >> v.w);
}

}

// _Vector4D namespace end

//! This class provides the function to check the correctness of the class Vector4D
class Vector4DTest
{
public:
    Vector4DTest() { }
    ~Vector4DTest() { }

    //! A static function which tests Vector4D class for correctness
    template<class T> static void runf()
    {
        const int numberOfTests = 48;
        int passed = 0;

        cout << endl << "!Test started....." << endl << endl;

        /// Constructors ---------->

        _Vector4D::Vector4D<T> v_1;

        if( fabs(v_1.x) <= numeric_limits<T>::epsilon() && fabs(v_1.y) <= numeric_limits<T>::epsilon() && fabs(v_1.z) <= numeric_limits<T>::epsilon() && fabs(v_1.w) <= numeric_limits<T>::epsilon() ) {
            cout << "Default constructor test: PASSED." << endl;
            passed++;
        } else cout << "Default constructor test: FAILED." << endl;

        _Vector4D::Vector4D<T> v_2_1(1.0);

        if( fabs(v_2_1.x - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_2_1.y - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_2_1.z - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_2_1.w - 1.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constructor with one explicit coordinate test 1: PASSED." << endl;
            passed++;
        } else cout << "Constructor with one explicit coordinate test 1: FAILED." << endl;

        T scalar = 3.0;
        _Vector4D::Vector4D<T> v_2_2 = scalar;

        if( fabs(v_2_2.x - 3.0) <= numeric_limits<T>::epsilon() && fabs(v_2_2.y - 3.0) <= numeric_limits<T>::epsilon() && fabs(v_2_2.z - 3.0) <= numeric_limits<T>::epsilon() && fabs(v_2_2.w - 3.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constructor with one explicit coordinate test 2: PASSED." << endl;
            passed++;
        } else cout << "Constructor with one explicit coordinate test 2: FAILED." << endl;

        _Vector4D::Vector4D<T> v_3(v_2_1.x, 2.0);

        if( fabs(v_3.x - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_3.y - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_3.z - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_3.w - 2.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constructor with two explicit coordinates test: PASSED." << endl;
            passed++;
        } else cout << "Constructor with two explicit coordinates test: FAILED." << endl;

        _Vector4D::Vector4D<T> v_4(v_2_1.x, 2.0, 1.0);

        if( fabs(v_4.x - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_4.y - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_4.z - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_4.w - 1.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constructor with three explicit coordinates test: PASSED." << endl;
            passed++;
        } else cout << "Constructor with three explicit coordinates test: FAILED." << endl;

        _Vector4D::Vector4D<T> v_4_0(v_2_1.x, 2.0, 1.0, v_4.y);

        if( fabs(v_4_0.x - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_4_0.y - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_4_0.z - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_4_0.w - 2.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constructor with four explicit coordinates test: PASSED." << endl;
            passed++;
        } else cout << "Constructor with four explicit coordinates test: FAILED." << endl;

        _Vector4D::Vector4D<T> v_4_1(v_4);

        if( fabs(v_4_1.x - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_4_1.y - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_4_1.z - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_4_1.w - 1.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Copy constructor test 1: PASSED." << endl;
            passed++;
        } else cout << "Copy constructor test 1: FAILED." << endl;

        _Vector4D::Vector4D<T> v_4_2 = v_4_1;

        if( fabs(v_4_2.x - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_4_2.y - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_4_2.z - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_4_2.w - 1.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Copy constructor test 2: PASSED." << endl;
            passed++;
        } else cout << "Copy constructor test 2: FAILED." << endl;

        T a[4] = { 2.0, 2.0, 2.0, 2.0 };
        _Vector4D::Vector4D<T> v_5_1(a);

        if( fabs(v_5_1.x - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_5_1.y - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_5_1.z - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_5_1.w - 2.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constructor from an array test 1: PASSED." << endl;
            passed++;
        } else cout << "Constructor from an array test 1: FAILED." << endl;

        _Vector4D::Vector4D<T> v_5_2 = a;

        if( fabs(v_5_2.x - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_5_2.y - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_5_2.z - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_5_2.w - 2.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constructor from an array test 2: PASSED." << endl;
            passed++;
        } else cout << "Constructor from an array test 2: FAILED." << endl;

        vector<T> v(4);
        v[0] = 1.0; v[1] = 2.0; v[2] = 1.0; v[3] = 2.0;
        _Vector4D::Vector4D<T> v_6_1(v);

        if( fabs(v_6_1[0] - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_6_1[1] - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_6_1[2] - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_6_1[3] - 2.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constructor from a standard C++ vector test 1: PASSED." << endl;
            passed++;
        } else cout << "Constructor from a standard C++ vector test 1: FAILED." << endl;

        _Vector4D::Vector4D<T> v_6_2 = v;

        if( fabs(v_6_2[0] - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_6_2[1] - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_6_2[2] - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_6_2[3] - 2.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constructor from a standard C++ vector test 2: PASSED." << endl;
            passed++;
        } else cout << "Constructor from a standard C++ vector test 2: FAILED." << endl;

        /// Operator [] ---------->

        const _Vector4D::Vector4D<T> v_7(1.0, 2.0, 3.0, 4.0);

        if( fabs(v_7[0] - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_7[1] - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_7[2] - 3.0) <= numeric_limits<T>::epsilon() && fabs(v_7[3] - 4.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Constant object []-operator test: PASSED." << endl;
            passed++;
        } else cout << "Constant object []-operator test: FAILED." << endl;

        _Vector4D::Vector4D<T> v_8(1.0, 2.0, 3.0, 4.0);
        T sum = 0.0;
        for(int i = 0; i < 4; ++i) sum += v_8[i];

        if( fabs(sum - 10.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Manipulation []-operator test: PASSED." << endl;
            passed++;
        } else cout << "Manipulation []-operator test: FAILED." << endl;

        /// Operator = ---------->

        _Vector4D::Vector4D<T> v_9;
        v_9 = scalar;

        if( fabs(v_9.x - 3.0) <= numeric_limits<T>::epsilon() && fabs(v_9.y - 3.0) <= numeric_limits<T>::epsilon() && fabs(v_9.z - 3.0) <= numeric_limits<T>::epsilon() && fabs(v_9.w - 3.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Operator=(scalar) test: PASSED." << endl;
            passed++;
        } else cout << "Operator=(scalar) test: FAILED." << endl;

        _Vector4D::Vector4D<T> v_10;
        v_10 = v_9;

        if( fabs(v_10.x - 3.0) <= numeric_limits<T>::epsilon() && fabs(v_10.y - 3.0) <= numeric_limits<T>::epsilon() && fabs(v_10.z - 3.0) <= numeric_limits<T>::epsilon() && fabs(v_10.w - 3.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Operator=(Vector4D) test: PASSED." << endl;
            passed++;
        } else cout << "Operator=(Vector4D) test: FAILED." << endl;

        _Vector4D::Vector4D<T> v_11;
        v_11 = a;

        if( fabs(v_11.x - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_11.y - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_11.z - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_11.w - 2.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Operator=(C++ array) test: PASSED." << endl;
            passed++;
        } else cout << "Operator=(C++ array) test: FAILED." << endl;

        _Vector4D::Vector4D<T> v_12;
        v_12 = v;

        if( fabs(v_12.x - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_12.y - 2.0) <= numeric_limits<T>::epsilon() && fabs(v_12.z - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_12.w - 2.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Operator=(C++ vector) test: PASSED." << endl;
            passed++;
        } else cout << "Operator=(C++ vector) test: FAILED." << endl;

        /// Operators == and != ---------->

        _Vector4D::Vector4D<T> v_13_1(1.0, 2.0, 3.0, 4.0);
        _Vector4D::Vector4D<T> v_13_2(1.0, 3.0, 4.0, 2.0);

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

        _Vector4D::Vector4D<T> v_14(1.0, 1.0, 1.0, 1.0);
        _Vector4D::Vector4D<T> v_15(2.0, 2.0, 2.0, 2.0);
        _Vector4D::Vector4D<T> v_res(3.0, 3.0, 3.0, 3.0);

        if( (v_14 + v_15) ==  v_res ) {
            cout << "Operator + test: PASSED." << endl;
            passed++;
        } else cout << "Operator + test: FAILED." << endl;

        v_res = _Vector4D::Vector4D<T>(-1.0, -1.0, -1.0, -1.0);

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

        v_res = _Vector4D::Vector4D<T>(3.0, 3.0, 3.0, 3.0);

        v_14 = _Vector4D::Vector4D<T>(1.0, 1.0, 1.0, 1.0);
        v_14 += v_15;
        if( v_14 ==  v_res ) {
            cout << "Operator += test: PASSED." << endl;
            passed++;
        } else cout << "Operator += test: FAILED." << endl;

        /// Operators / and * ---------->

        v_14 = _Vector4D::Vector4D<T>(1.0, 1.0, 1.0, 1.0);

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

        v_14 = _Vector4D::Vector4D<T>(1.0, 1.0, 1.0, 1.0);
        v_res = _Vector4D::Vector4D<T>(0.5, 0.5, 0.5, 0.5);

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

        v_14 = _Vector4D::Vector4D<T>(1.0, 1.0, 1.0, 1.0);

        if( fabs(v_14*v_15 - 8.0) <= numeric_limits<T>::epsilon() && fabs(v_14.scalarProduct(v_15) - 8.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Scalar product test: PASSED." << endl;
            passed++;
        } else cout << "Scalar product test: FAILED." << endl;

        /// Norms ---------->

        v_15 = _Vector4D::Vector4D<T>(0.0, 0.0, 3.0, 4.0);

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

        /// The smallest and the biggest coordinates ---------->

        v_15 = _Vector4D::Vector4D<T>(1.0, 0.0, 3.0, 4.0);

        if( v_15.indexOfTheSmallestCoordinate() == 1) {
            cout << "Index of the smallest coordinate test: PASSED." << endl;
            passed++;
        } else cout << "Index of the smallest coordinate test: FAILED." << endl;

        if( fabs(v_15.theSmallestCoordinate()) <= numeric_limits<T>::epsilon() ) {
            cout << "The smallest coordinate test: PASSED." << endl;
            passed++;
        } else cout << "The smallest coordinate test: FAILED." << endl;

        v_15 = _Vector4D::Vector4D<T>(1.0, 0.0, 4.0, 3.0);

        if( v_15.indexOfTheBiggestCoordinate() == 2) {
            cout << "Index of the biggest coordinate test: PASSED." << endl;
            passed++;
        } else cout << "Index of the biggest coordinate test: FAILED." << endl;

        if( fabs(v_15.theBiggestCoordinate() - 4.0) <= numeric_limits<T>::epsilon() ) {
            cout << "The biggest coordinate test: PASSED." << endl;
            passed++;
        } else cout << "The biggest coordinate test: FAILED." << endl;

        /// Cartesianization ---------->

        v_15 = _Vector4D::Vector4D<T>(2.0, -2.0, 2.0, 4.0);

        v_15.cartesianize();
        if( fabs(v_15.x - 0.5) <= numeric_limits<T>::epsilon() && fabs(v_15.y + 0.5) <= numeric_limits<T>::epsilon() && fabs(v_15.z - 0.5) <= numeric_limits<T>::epsilon() && fabs(v_15.w - 1.0) <= numeric_limits<T>::epsilon() ) {
            cout << "Cartesianize test: PASSED." << endl;
            passed++;
        } else cout << "Cartesianize test: FAILED." << endl;

        if( (_Vector4D::Vector4D<T>(2.0, -2.0, 2.0, 4.0).cartesianized() == _Vector4D::Vector4D<T>(0.5, -0.5, 0.5, 1.0)) == true) {
            cout << "Cartesianized test: PASSED." << endl;
            passed++;
        } else cout << "Cartesianized test: FAILED." << endl;

        /// Normalization ---------->

        v_15 = _Vector4D::Vector4D<T>(0.0, 0.0, 3.0, 4.0);

        v_15.normalize();
        if( fabs(v_15.x) <= numeric_limits<T>::epsilon() && fabs(v_15.y) <= numeric_limits<T>::epsilon() && fabs(v_15.z - 0.6) <= numeric_limits<T>::epsilon() && fabs(v_15.w - 0.8) <= numeric_limits<T>::epsilon() ) {
            cout << "Normalize test: PASSED." << endl;
            passed++;
        } else cout << "Normalize test: FAILED." << endl;

        if( (_Vector4D::Vector4D<T>(0.0, 0.0, 0.0, 0.0).normalized() == _Vector4D::Vector4D<T>(0.0, 0.0, 0.0, 1.0)) == true) {
            cout << "Normalized test: PASSED." << endl;
            passed++;
        } else cout << "Normalized test: FAILED." << endl;

        /// Reflection ---------->

        v_14 = _Vector4D::Vector4D<T>(-1.0, 0.0, 0.0, 0.0);
        v_15 = _Vector4D::Vector4D<T>(0.0, 0.0, 0.0, 1.0);

        v_14.reflect(v_15);
        if( fabs(v_14.x - 1.0) <= numeric_limits<T>::epsilon() && fabs(v_14.y) <= numeric_limits<T>::epsilon() && fabs(v_14.z) <= numeric_limits<T>::epsilon() && fabs(v_14.w) <= numeric_limits<T>::epsilon() ) {
            cout << "Reflect test: PASSED." << endl;
            passed++;
        } else cout << "Reflect test: FAILED." << endl;

        if( (_Vector4D::Vector4D<T>(-1.0, 0.0, 0.0, 0.0).reflected(v_15) == _Vector4D::Vector4D<T>(1.0, 0.0, 0.0, 0.0)) == true) {
            cout << "Reflected test: PASSED." << endl;
            passed++;
        } else cout << "Reflected test: FAILED." << endl;

        /// Absolute value ---------->

        v_14 = _Vector4D::Vector4D<T>(3.0, 4.0, 5.0, 6.0);
        v_15 = _Vector4D::Vector4D<T>(-3.0, -4.0, -5.0, -6.0);
        v_15.abs();
        if( fabs(v_15.x - v_14.x) <= numeric_limits<T>::epsilon() && fabs(v_15.y - v_14.y) <= numeric_limits<T>::epsilon() && fabs(v_15.z - v_14.z) <= numeric_limits<T>::epsilon() && fabs(v_15.w - v_14.w) <= numeric_limits<T>::epsilon() ) {
            cout << "Absolute value test: PASSED." << endl;
            passed++;
        } else cout << "Absolute value test: FAILED." << endl;

        cout << endl << ".....Test finished!" << endl;

        cout << endl;
        cout << "Overall test result: " << endl << endl;
        cout << "Number of tests = " << numberOfTests << ";" << endl;
        cout << "Tests passed: " << passed << "/" << numberOfTests << ";" << endl;
        cout << "Tests failed: " << numberOfTests - passed << "/" << numberOfTests << ";" << endl;

        if(numberOfTests == passed) cout << "TestVector4DClass: SUCCESS!" << endl;
        else cout << "TestVector4DClass: FAILED!" << endl;
    }
};

#endif // Vector4D_H
