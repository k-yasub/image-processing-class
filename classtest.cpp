#include <cmath>
#include <iostream>

namespace myclass {
class complex {  // example class for complex numbers
 private:
  double real;  // real part
  double imag;  // imaginary part
 public:
  complex() : real(-1.0), imag(-1.0) {  // Constructor
  }
  complex(double a, double b) : real(a), imag(b){};
  double get_real() const { return real; }
  double get_imag() const { return imag; }
  double abs() { return sqrt(real * real + imag * imag); }

  complex operator+(const complex& b) {
    double r = this->get_real() + b.get_real();
    double i = this->get_imag() + b.get_imag();
    complex tmp(r, i);
    return tmp;
  }

  complex operator-(const complex& b) {
    double r = this->get_real() - b.get_real();
    double i = this->get_imag() - b.get_imag();
    complex tmp(r, i);
    return tmp;
  }
};

}  // namespace myclass

int main() {
  // C++のクラスインスタンス
  myclass::complex* a;
  myclass::complex b(5, -7);
  myclass::complex c(1, 1);
  myclass::complex d = b + c;
  a = &b;
  std::cout << "a = " << a->abs() << std::endl;
  std::cout << "b = " << b.abs() << std::endl;
}