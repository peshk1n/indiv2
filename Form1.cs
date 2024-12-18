using System;
using System.Drawing;
using System.Numerics;
using static indiv02.Form1;

namespace indiv02
{
    public partial class Form1 : Form
    {
        public List<Triangle> triangles;
        public List<Sphere> spheres;
        public List<Light> lights;
        public Camera camera;

        //public TransformationMatrix projectionMatrix;


        // Класс точки
        // -----------
        public class Point3D
        {
            public double X { get; set; }
            public double Y { get; set; }
            public double Z { get; set; }

            public Point3D(double x, double y, double z)
            {
                X = x;
                Y = y;
                Z = z;
            }

            public static Point3D Vector(Point3D a, Point3D b)
            {
                return new Point3D(b.X - a.X, b.Y - a.Y, b.Z - a.Z);
            }

            // Метод для скалярного произведения
            public static double DotProduct(Point3D a, Point3D b)
            {
                return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
            }

            // Метод для вычисления скалярного произведения
            public double Dot(Point3D other)
            {
                return this.X * other.X + this.Y * other.Y + this.Z * other.Z;
            }

            public Point3D Cross(Point3D other)
            {
                return new Point3D(
                    Y * other.Z - Z * other.Y,
                    Z * other.X - X * other.Z,
                    X * other.Y - Y * other.X
                );
            }


            // Метод для векторного произведения
            public static Point3D CrossProduct(Point3D a, Point3D b)
            {
                return new Point3D(
                    a.Y * b.Z - a.Z * b.Y,
                    a.Z * b.X - a.X * b.Z,
                    a.X * b.Y - a.Y * b.X
                );
            }

            // Метод для нормализации вектора (приведение к единичной длине)
            public Point3D Normalize()
            {
                double length = Math.Sqrt(X * X + Y * Y + Z * Z);
                return new Point3D(X / length, Y / length, Z / length);
            }

            public double Length()
            {
                return Math.Sqrt(X * X + Y * Y + Z * Z);
            }

            // Пример перегрузки
            public static Point3D operator +(Point3D a, Point3D b) =>
                new Point3D(a.X + b.X, a.Y + b.Y, a.Z + b.Z);

            public static Point3D operator -(Point3D a, Point3D b) =>
                new Point3D(a.X - b.X, a.Y - b.Y, a.Z - b.Z);

            public static Point3D operator *(Point3D a, double scalar) =>
                new Point3D(a.X * scalar, a.Y * scalar, a.Z * scalar);

            public static Point3D operator /(Point3D a, double scalar) =>
                new Point3D(a.X / scalar, a.Y / scalar, a.Z / scalar);

            public static Point3D operator -(Point3D point) =>
                new Point3D(-point.X, -point.Y, -point.Z);

        }


        // Класс грани
        // -----------
        public class Face
        {
            public List<int> Vertices { get; set; }
            public Material FaceMaterial { get; set; }
            public Point3D Normal { get; set; }

            public Face(List<int> vertices)
            {
                Vertices = vertices;
                FaceMaterial = new Material(
                    diffuseColor: Color.White,  // Чисто белый цвет
                    reflectivity: 0.0,          // Нет зеркальных отражений
                    refractiveIndex: 0.0        // Непрозрачная поверхность, без преломления
                );
            }
        }

        // Класс многогранника
        // -------------------
        public class Polyhedron
        {
            public List<Point3D> Vertices { get; set; } //точки
            public List<Face> Faces { get; set; } //грани

            public Polyhedron()
            {
                Vertices = new List<Point3D>();
                Faces = new List<Face>();
            }

            public Polyhedron(List<Point3D> vertices, List<Face> faces)
            {
                Vertices = vertices;
                Faces = faces;
            }

            public void ApplyTransformation(TransformationMatrix matrix)
            {
                for (int i = 0; i < Vertices.Count; i++)
                {
                    Vertices[i] = matrix.Transform(Vertices[i]);
                }
            }


            // Метод для вычисления нормалей каждой грани
            public void CalculateNormals()
            {
                foreach (var face in Faces)
                {
                    if (face.Vertices.Count >= 3) // Требуется минимум 3 вершины для определения нормали
                    {
                        // Получаем векторы двух сторон грани
                        var vector1 = Point3D.Vector(Vertices[face.Vertices[0]], Vertices[face.Vertices[1]]);
                        var vector2 = Point3D.Vector(Vertices[face.Vertices[0]], Vertices[face.Vertices[2]]);

                        // Вычисляем нормаль как векторное произведение двух векторов
                        //face.Normal = Point3D.CrossProduct(vector1, vector2);
                        face.Normal = Point3D.CrossProduct(vector1, vector2).Normalize();
                    }
                    else
                    {
                        face.Normal = new Point3D(0, 0, 0); // Если не хватает точек, нормаль по умолчанию
                    }
                }
            }
        }

        // Класс источника света 
        // ---------------------
        public class Light
        {
            public Point3D Position { get; set; }
            public double Intensity { get; set; }
            public Color Color { get; set; }
            public Light(Point3D position, double intensity, Color color)
            {
                Position = position;
                Color = color;
                Intensity = intensity;
            }
        }

        // Класс камеры
        // ------------
        public class Camera
        {
            public Point3D Position { get; set; }
            public Point3D Direction { get; set; }

            public Camera(Point3D position, Point3D direction)
            {
                Position = position;
                Direction = direction.Normalize(); ;
            }

            public Point3D GetDirection()
            {
                return new Point3D(0, 0, 1);
            }
        }


        // Класс матрицы афинных преобразований
        // ------------------------------------
        public class TransformationMatrix
        {
            public double[,] matrix = new double[4, 4];

            public TransformationMatrix()
            {
                matrix[0, 0] = 1;
                matrix[1, 1] = 1;
                matrix[2, 2] = 1;
                matrix[3, 3] = 1;
            }


            public Point3D Transform(Point3D p)
            {
                double x = matrix[0, 0] * p.X + matrix[1, 0] * p.Y + matrix[2, 0] * p.Z + matrix[3, 0];
                double y = matrix[0, 1] * p.X + matrix[1, 1] * p.Y + matrix[2, 1] * p.Z + matrix[3, 1];
                double z = matrix[0, 2] * p.X + matrix[1, 2] * p.Y + matrix[2, 2] * p.Z + matrix[3, 2];

                return new Point3D(x, y, z);
            }

            public Triangle TransformTriangle(Triangle triangle)
            {
                return new Triangle(Transform(triangle.A), Transform(triangle.B), Transform(triangle.C), triangle.Material);
            }


            public Point3D TransformForPerspect(Point3D p)
            {
                double x = matrix[0, 0] * p.X + matrix[1, 0] * p.Y + matrix[2, 0] * p.Z + matrix[3, 0];
                double y = matrix[0, 1] * p.X + matrix[1, 1] * p.Y + matrix[2, 1] * p.Z + matrix[3, 1];
                double z = matrix[0, 2] * p.X + matrix[1, 2] * p.Y + matrix[2, 2] * p.Z + matrix[3, 2];
                double w = matrix[0, 3] * p.X + matrix[1, 3] * p.Y + matrix[2, 3] * p.Z + matrix[3, 3];

                if (w != 0)
                {
                    x /= w;
                    y /= w;
                    z /= w;
                }

                return new Point3D(x, y, z);
            }

            public static TransformationMatrix PerspectiveProjection(double distance)
            {
                // Перспективная проекция
                TransformationMatrix result = new TransformationMatrix();

                result.matrix[0, 0] = 1;
                result.matrix[1, 1] = 1;
                result.matrix[3, 2] = 1 / distance;
                result.matrix[3, 3] = 1;
                return result;
            }

            public static TransformationMatrix AxonometricProjection(double theta, double phi)
            {
                // преобразуем углы из градусов в радианы
                double cosTheta = Math.Cos(theta * Math.PI / 180);
                double sinTheta = Math.Sin(theta * Math.PI / 180);
                double cosPhi = Math.Cos(phi * Math.PI / 180);
                double sinPhi = Math.Sin(phi * Math.PI / 180);

                TransformationMatrix result = new TransformationMatrix();
                result.matrix[0, 0] = cosTheta;
                result.matrix[0, 1] = sinTheta * sinPhi;
                result.matrix[1, 1] = cosPhi;
                result.matrix[2, 0] = sinTheta;
                result.matrix[2, 1] = -cosTheta * sinPhi;
                result.matrix[2, 2] = 0;
                return result;
            }

            public static TransformationMatrix Multiply(TransformationMatrix a, TransformationMatrix b)
            {
                TransformationMatrix result = new TransformationMatrix();
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        result.matrix[i, j] = 0;
                        for (int k = 0; k < 4; k++)
                        {
                            result.matrix[i, j] += a.matrix[i, k] * b.matrix[k, j];
                        }
                    }
                }
                return result;
            }

            public static TransformationMatrix CreateTranslationMatrix(double dx, double dy, double dz)
            {
                TransformationMatrix result = new TransformationMatrix();
                result.matrix[3, 0] = dx;
                result.matrix[3, 1] = dy;
                result.matrix[3, 2] = dz;
                return result;
            }

            public static TransformationMatrix CreateScalingMatrix(double scaleX, double scaleY, double scaleZ)
            {
                TransformationMatrix result = new TransformationMatrix();
                result.matrix[0, 0] = scaleX;
                result.matrix[1, 1] = scaleY;
                result.matrix[2, 2] = scaleZ;
                return result;
            }

            public static TransformationMatrix CreateScalingMatrix(double scaleX, double scaleY, double scaleZ, Point3D center)
            {
                var translateToOrigin = CreateTranslationMatrix(-center.X, -center.Y, -center.Z);
                var scalingMatrix = CreateScalingMatrix(scaleX, scaleY, scaleZ);
                var translateBack = CreateTranslationMatrix(center.X, center.Y, center.Z);

                return Multiply(Multiply(translateToOrigin, scalingMatrix), translateBack);
            }


            public static TransformationMatrix CreateRotationMatrixX(double angle)
            {
                double radians = angle * Math.PI / 180;
                TransformationMatrix result = new TransformationMatrix();
                double cos = Math.Cos(radians);
                double sin = Math.Sin(radians);
                result.matrix[1, 1] = cos;
                result.matrix[1, 2] = -sin;
                result.matrix[2, 1] = sin;
                result.matrix[2, 2] = cos;
                return result;
            }

            public static TransformationMatrix CreateRotationMatrixY(double angle)
            {
                double radians = angle * Math.PI / 180;
                TransformationMatrix result = new TransformationMatrix();
                double cos = Math.Cos(radians);
                double sin = Math.Sin(radians);
                result.matrix[0, 0] = cos;
                result.matrix[0, 2] = -sin;
                result.matrix[2, 0] = sin;
                result.matrix[2, 2] = cos;
                return result;
            }

            public static TransformationMatrix CreateRotationMatrixY(double angle, Point3D center)
            {
                TransformationMatrix translateToOrigin = CreateTranslationMatrix(-center.X, -center.Y, -center.Z);
                TransformationMatrix rotationMatrix = CreateRotationMatrixY(angle);
                TransformationMatrix translateBack = CreateTranslationMatrix(center.X, center.Y, center.Z);
                return Multiply(Multiply(translateToOrigin, rotationMatrix), translateBack);
            }

            public static TransformationMatrix CreateRotationMatrixZ(double angle)
            {
                double radians = angle * Math.PI / 180;
                TransformationMatrix result = new TransformationMatrix();
                double cos = Math.Cos(radians);
                double sin = Math.Sin(radians);
                result.matrix[0, 0] = cos;
                result.matrix[0, 1] = sin;
                result.matrix[1, 0] = -sin;
                result.matrix[1, 1] = cos;
                return result;
            }

            public static TransformationMatrix CreateReverseRotationMatrix(double angleX, double angleY, double angleZ)
            {
                var rotationX = CreateRotationMatrixX(angleX);
                var rotationY = CreateRotationMatrixY(angleY);
                var rotationZ = CreateRotationMatrixZ(angleZ);
                return Multiply(Multiply(rotationZ, rotationY), rotationX);
            }

            public static TransformationMatrix CreateReverseRotationMatrix(double angleX, double angleY, double angleZ, Point3D center)
            {
                TransformationMatrix translateToOrigin = CreateTranslationMatrix(-center.X, -center.Y, -center.Z);
                TransformationMatrix rotationMatrix = CreateReverseRotationMatrix(angleX, angleY, angleZ);
                TransformationMatrix translateBack = CreateTranslationMatrix(center.X, center.Y, center.Z);
                return Multiply(Multiply(translateToOrigin, rotationMatrix), translateBack);
            }

            public static TransformationMatrix CreateRotationMatrix(double angleX, double angleY, double angleZ)
            {
                var rotationX = CreateRotationMatrixX(angleX);
                var rotationY = CreateRotationMatrixY(angleY);
                var rotationZ = CreateRotationMatrixZ(angleZ);
                return Multiply(Multiply(rotationX, rotationY), rotationZ);
            }

            public static TransformationMatrix CreateRotationAroundAxis(double angleX, double angleY, double angleZ, Point3D center)
            {
                TransformationMatrix translateToOrigin = CreateTranslationMatrix(-center.X, -center.Y, -center.Z);
                TransformationMatrix rotationMatrix = CreateRotationMatrix(angleX, angleY, angleZ);
                TransformationMatrix translateBack = CreateTranslationMatrix(center.X, center.Y, center.Z);
                return Multiply(Multiply(translateToOrigin, rotationMatrix), translateBack);
            }
        }

        // Структуры для алгоритма обратной трассировки лучей
        // --------------------------------------------------
        // Примитивы для создания сцены 
        // ----------------------------
        public class Triangle
        {
            public Point3D A { get; }
            public Point3D B { get; }
            public Point3D C { get; }
            public Point3D Normal { get; set; }
            public Material Material { get; }

            public Triangle(Point3D a, Point3D b, Point3D c, Material material)
            {
                A = a;
                B = b;
                C = c;
                Material = material;
                CalcNormal();
            }

            // Метод для нахождения нормали треугольника
            public void CalcNormal()
            {
                Point3D edge1 = B - A;
                Point3D edge2 = C - A;
                Normal = Point3D.CrossProduct(edge1, edge2).Normalize();
            }

            // Метод для проверки пересечения луча с треугольником
            // алгоритм Мёллера-Трумбора
            public bool Intersect(Ray ray, out double t)
            {
                t = 0;
                double epsilon = 1e-4f;

                Point3D edge1 = B - A;
                Point3D edge2 = C - A;
                Point3D h = Point3D.CrossProduct(ray.Direction, edge2);
                double a = Point3D.DotProduct(edge1, h);

                // проверяем, параллелен ли луч плоскости
                if (Math.Abs(a) < epsilon) 
                    return false;

                // вычисляем барицентрические координаты
                double f = 1.0f / a;
                Point3D s = ray.Origin - A;
                double u = f * Point3D.DotProduct(s, h);

                // находится ли точка вне треугольника по одной из осей
                if (u < 0.0f || u > 1.0f) 
                    return false;

                Point3D q = Point3D.CrossProduct(s, edge1);
                double v = f * Point3D.DotProduct(ray.Direction, q);

                // проверяем, что точка пересечения внутри треугольника
                if (v < 0.0f || u + v > 1.0f) 
                    return false;

                t = f * Point3D.DotProduct(edge2, q);

                return t > epsilon; 
            }
        }

        public class Sphere
        {
            public Point3D Center { get; set; }
            public double Radius { get; set; }
            public Material Material { get; set; }

            public Sphere(Point3D center, double radius, Material material)
            {
                Center = center;
                Radius = radius;
                Material = material;
            }


            // Метод для проверки пересечения луча и сферы
            // уравнение пересечения луча и сферы
            public bool Intersect(Ray ray, out double t)
            {
                t = 0;

                Point3D oc = ray.Origin - Center;

                double a = Point3D.DotProduct(ray.Direction, ray.Direction);
                double b = 2.0f * Point3D.DotProduct(oc, ray.Direction);
                double c = Point3D.DotProduct(oc, oc) - Radius * Radius;

                double D = b * b - 4 * a * c;

                if (D < 0)
                    return false; // Нет пересечений

                double sqrtD = (float)Math.Sqrt(D);
                double t1 = (-b - sqrtD) / (2.0f * a);
                double t2 = (-b + sqrtD) / (2.0f * a);

                if (t1 > 0)
                {
                    t = t1;
                }
                else if (t2 > 0)
                {
                    t = t2; 
                }
                else
                {
                    return false; 
                }

                return true; 
            }
        }

        // Класс материала
        // ---------------
        public class Material
        {
            public Color DiffuseColor { get; set; }
            public double Reflectivity { get; set; }
            public double Refractivity { get; set; } 
            public double RefractiveIndex { get; set; }
            public double Shininess { get; set; }

            public Material(Color diffuseColor, double reflectivity, double refractiveIndex, double refractivity = 0.0, double shininess = 100.0)
            {
                DiffuseColor = diffuseColor;
                Reflectivity = reflectivity;
                Refractivity = refractivity;
                RefractiveIndex = refractiveIndex;
                Shininess = shininess;
            }
        }

        // Класс луча
        // ----------
        public class Ray
        {
            public Point3D Origin { get; set; } // начальная точка луча
            public Point3D Direction { get; set; } // направление

            public Ray(Point3D origin, Point3D direction)
            {
                Origin = origin;
                Direction = direction.Normalize();
            }

            public Point3D GetPoint(double t)
            {
                return Origin + Direction * t; // Точка на луче на расстоянии t
            }

            public Ray Reflect(Point3D hitPoint, Point3D normal)
            {
                // Направление отражённого луча: R = D - 2 * (D · N) * N
                Point3D reflectedDirection = (Direction - normal * 2 * Point3D.DotProduct(Direction, normal)).Normalize();
                const double epsilon = 1e-6;
                Point3D origin = hitPoint + reflectedDirection * epsilon;
                return new Ray(origin, reflectedDirection);
            }

            public Ray? Refract(Point3D hitPoint, Point3D normal, double refractiveIndex)
            {
                normal = normal.Normalize();
                Direction = Direction.Normalize();

                double cosI = -Math.Max(-1.0, Math.Min(1.0, Direction.Dot(normal)));

                // Показатели преломления внешней среды и материала
                double n1 = 1.0;
                double n2 = refractiveIndex;

                if (cosI < 0)
                {
                    cosI = -cosI;
                    normal = -normal;
                    (n1, n2) = (n2, n1);
                }

                double eta = n1 / n2;
                // Проверка полного внутреннего отражения (когда луч полностью отражается и не проникает во вторую среду)
                double k = 1 - eta * eta * (1 - cosI * cosI);
                if (k < 0)
                {
                    return null;
                }

                Point3D refractedDirection = Direction * eta + normal * (eta * cosI - Math.Sqrt(k));
                Point3D offset = refractedDirection * 1e-7;
                // Возвращаем новый преломленный луч
                return new Ray(hitPoint + offset, refractedDirection.Normalize());
            }

        }


        // Методы для создания сцены 
        // -------------------------
        public void CreateCornellBox()
        {
            // --- Материалы ---
            Material floorMaterial = new Material(Color.White, 0.8, 0.0, 0.0, 0.0);
            Material ceilingMaterial = new Material(Color.White, 0.0, 0.0, 0.0, 0.0); 
            Material leftWallMaterial = new Material(Color.Red, 0.0, 0.0, 0.0, 0.0); 
            Material rightWallMaterial = new Material(Color.SteelBlue, 0.0, 0.0, 0.0, 0.0); 
            Material backWallMaterial = new Material(Color.White, 0.0, 0.0, 0.0, 0.0); 
            Material frontWallMaterial = new Material(Color.White, 0.0, 0.0, 0.0, 0.0); 

            List<Triangle> room = new List<Triangle>();

            // --- Вершины ---
            Point3D p1 = new Point3D(-1, -1, -1);
            Point3D p2 = new Point3D(1, -1, -1);
            Point3D p3 = new Point3D(1, -1, 1);
            Point3D p4 = new Point3D(-1, -1, 1);
            Point3D p5 = new Point3D(-1, 1, -1);
            Point3D p6 = new Point3D(1, 1, -1);
            Point3D p7 = new Point3D(1, 1, 1);
            Point3D p8 = new Point3D(-1, 1, 1);

            // Пол
            room.Add(new Triangle(p1, p4, p2, floorMaterial));
            room.Add(new Triangle(p4, p3, p2, floorMaterial));

            // Потолок
            room.Add(new Triangle(p5, p6, p8, ceilingMaterial));
            room.Add(new Triangle(p6, p7, p8, ceilingMaterial));

            // Левая стена (красная)
            room.Add(new Triangle(p1, p5, p4, leftWallMaterial));
            room.Add(new Triangle(p5, p8, p4, leftWallMaterial));

            // Правая стена (зелёная)
            room.Add(new Triangle(p2, p3, p6, rightWallMaterial));
            room.Add(new Triangle(p3, p7, p6, rightWallMaterial));

            // Задняя стена
            room.Add(new Triangle(p1, p2, p5, backWallMaterial));
            room.Add(new Triangle(p2, p6, p5, backWallMaterial));

            // Передняя стена
            room.Add(new Triangle(p4, p8, p3, frontWallMaterial));
            room.Add(new Triangle(p8, p7, p3, frontWallMaterial));

            triangles.AddRange(room);
        }

        private void TransformFigure(List<Triangle> fig, TransformationMatrix matrix)
        {
            for (int i = 0; i < fig.Count; i++)
            {
                fig[i] = matrix.TransformTriangle(fig[i]);
                fig[i].CalcNormal();
            }
        }

        private void BuildScene()
        {
            // --- Добавление комнаты ---
            CreateCornellBox();

            // --- Материалы ---
            Material sphereMaterial1 = new Material(Color.SeaGreen, 0.5, 0.0);
            Material sphereMaterial2 = new Material(Color.SkyBlue, 0.0, 0.0);

            Material cubeMaterial1 = new Material(Color.Goldenrod, 0.5, 0.0);
            Material cubeMaterial2 = new Material(Color.Sienna, 0.0, 0.0);
            Material cubeMaterial3 = new Material(Color.SlateGray, 0.0, 0.0);
            Material glassMaterial = new Material(Color.White, 0.0, 0.9, 0.5); // Белое стекло

            // --- Добавление сфер ---
            Sphere sphere = new Sphere(new Point3D(0.5, -0.7, 0.33), 0.3, sphereMaterial1);
            spheres.Add(sphere);
            Sphere sphere2 = new Sphere(new Point3D(0.0, -0.35, 0.8), 0.25, sphereMaterial2);
            spheres.Add(sphere2);

            // --- Добавление кубов ---
            Point3D cubeCenter = new Point3D(-0.6, -0.7, 0.6);
            var cube = CreateCube(cubeCenter, 0.6, cubeMaterial1);
            //TransformFigure(cube, TransformationMatrix.CreateRotationMatrixY(-25, cubeCenter));
            triangles.AddRange(cube);
            Point3D cubeCenter2 = new Point3D(-0.6, -0.25, 0.6);
            var cube2 = CreateCube(cubeCenter2, 0.3, cubeMaterial2);
            TransformFigure(cube2, TransformationMatrix.CreateRotationMatrixY(45, cubeCenter2));
            triangles.AddRange(cube2);
            Point3D cubeCenter3 = new Point3D(0.0, -0.8, 0.8);
            var cube3 = CreateCube(cubeCenter3, 0.4, cubeMaterial3);
            //TransformFigure(cube3, TransformationMatrix.CreateRotationMatrixY(45, cubeCenter3));
            triangles.AddRange(cube3);
            Point3D cubeCenter4 = new Point3D(0.4, -0.8, 0.8);
            var cube4 = CreateCube(cubeCenter4, 0.4, cubeMaterial3);
            triangles.AddRange(cube4);
            Point3D cubeCenter5 = new Point3D(0.8, -0.8, 0.8);
            var cube5 = CreateCube(cubeCenter5, 0.4, cubeMaterial3);
            triangles.AddRange(cube5);

            // --- Добавление источников света --- 
            lights.Add(new Light(new Point3D(0, 0.5, 0.0), 0.5, Color.White));

            /* lights.Add(new Light(new Point3D(0.8, 0.8, 0.9), 0.3, Color.White));
             lights.Add(new Light(new Point3D(0.0, 0.8, -0.6), 0.4, Color.White));*/

            //lights.Add(new Light(new Point3D(0.5, -0.5, -0.5), 0.4, Color.Purple));
            //lights.Add(new Light(new Point3D(-0.5, -0.5, -0.5), 0.4, Color.Yellow));
        }

        public List<Triangle> CreateCube(Point3D center, double size, Material material)
        {
            double halfSize = size / 2.0;

            // --- Вершины куба ---
            Point3D v0 = new Point3D(center.X - halfSize, center.Y - halfSize, center.Z - halfSize);
            Point3D v1 = new Point3D(center.X + halfSize, center.Y - halfSize, center.Z - halfSize);
            Point3D v2 = new Point3D(center.X + halfSize, center.Y + halfSize, center.Z - halfSize);
            Point3D v3 = new Point3D(center.X - halfSize, center.Y + halfSize, center.Z - halfSize);
            Point3D v4 = new Point3D(center.X - halfSize, center.Y - halfSize, center.Z + halfSize);
            Point3D v5 = new Point3D(center.X + halfSize, center.Y - halfSize, center.Z + halfSize);
            Point3D v6 = new Point3D(center.X + halfSize, center.Y + halfSize, center.Z + halfSize);
            Point3D v7 = new Point3D(center.X - halfSize, center.Y + halfSize, center.Z + halfSize);

            // --- Создание треугольников для каждой стороны куба ---
            List<Triangle> cube = new List<Triangle>
            {
                // Передняя грань
                new Triangle(v0, v2, v1, material),
                new Triangle(v0, v3, v2, material),

                // Задняя грань
                new Triangle(v4, v5, v6, material),
                new Triangle(v4, v6, v7, material),

                // Левая грань
                new Triangle(v0, v4, v7, material),
                new Triangle(v0, v7, v3, material),

                // Правая грань
                new Triangle(v1, v6, v5, material),
                new Triangle(v1, v2, v6, material),

                // Верхняя грань
                new Triangle(v3, v7, v6, material),
                new Triangle(v3, v6, v2, material),

                // Нижняя грань
                new Triangle(v0, v4, v5, material),
                new Triangle(v0, v5, v1, material)
            };

            return cube;
        }


        // Трассировка лучей
        // -----------------
        public Form1()
        {
            spheres = new List<Sphere>();
            lights = new List<Light>();
            triangles = new List<Triangle>();

            BuildScene();
            camera = new Camera(new Point3D(0, 0, -1), new Point3D(0, 0, 1));

            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            pictureBox1.Image = StartRayTrace();
        }

        public Bitmap StartRayTrace()
        {
            int width = pictureBox1.Width;
            int height = pictureBox1.Height;
            Bitmap bitmap = new Bitmap(width, height);

            // Область экрана для отображения (границы в мировых координатах)
            double screenWidth = 2.0; 
            double screenHeight = 2.0;
            double pixelWidth = screenWidth / width;
            double pixelHeight = screenHeight / height;

            Point3D right = new Point3D(1, 0, 0);
            Point3D up = new Point3D(0, 1, 0);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Рассчитываем смещение пикселя от центра экрана
                    double offsetX = (x + 0.5) * pixelWidth - screenWidth / 2;
                    double offsetY = screenHeight / 2 - (y + 0.5) * pixelHeight; 

                    Point3D rayDirection = (camera.Direction + right * offsetX + up * offsetY).Normalize();
                    Ray ray = new Ray(camera.Position, rayDirection);
                    Color pixelColor = RayTrace(ray, 5);
                    bitmap.SetPixel(x, y, pixelColor);
                }
            }

            return bitmap;
        }


        public Color RayTrace(Ray ray, int depth)
        {
            if (depth <= 0)
            {
                return Color.Black;
            }

            double closest_t = double.MaxValue;
            Point3D hitPoint = null;
            Point3D normal = null;
            Material? hitMaterial = null;

            // Проверка пересечения со сферами
            foreach (var sphere in spheres)
            {
                if (sphere.Intersect(ray, out double t) && t < closest_t)
                {
                    closest_t = t;
                    hitPoint = ray.GetPoint(t);
                    normal = (hitPoint - sphere.Center).Normalize();
                    hitMaterial = sphere.Material;
                }
            }

            // Проверка пересечения с треугольниками
            foreach (var triangle in triangles)
            {
                if (triangle.Intersect(ray, out double t) && t < closest_t)
                {
                    closest_t = t;
                    hitPoint = ray.GetPoint(t);
                    triangle.CalcNormal();
                    normal = triangle.Normal;
                    hitMaterial = triangle.Material;
                }
            }

            if (hitPoint == null)
            {
                return Color.Black;
            }

            // --- Рассчёт освещения ---
            Color resultColor = ApplyAmbientLighting(hitMaterial.DiffuseColor);
            foreach (var light in lights)
            {
                if (IsPointVisibleFromLight(hitPoint, light))
                {
                    Color localLighting = CalculateLocalLighting(hitMaterial, normal, hitPoint, light);
                    resultColor = AddColors(resultColor, localLighting);
                }
            }

            // --- Рассчёт отражения ---
            if (hitMaterial.Reflectivity > 0)
            {
                Ray reflectedRay = ray.Reflect(hitPoint, normal);
                Color reflectionColor = RayTrace(reflectedRay, depth - 1);
                resultColor = MixColors(resultColor, reflectionColor, hitMaterial.Reflectivity);
            }

            // --- Рассчёт преломления ---
            if (hitMaterial.Refractivity > 0)
            {
                Ray? refractedRay = ray.Refract(hitPoint, normal, hitMaterial.RefractiveIndex);
                if (refractedRay != null)
                {
                    Color refractionColor = RayTrace(refractedRay, depth - 1);
                    resultColor = MixColors(resultColor, refractionColor, hitMaterial.Refractivity);
                }
            }

            return resultColor;
        }


        private Color MixColors(Color local, Color reflection, double weight)
        {
            weight = Math.Max(0, Math.Min(1, weight));

            int r = (int)(local.R * (1 - weight) + reflection.R * weight);
            int g = (int)(local.G * (1 - weight) + reflection.G * weight);
            int b = (int)(local.B * (1 - weight) + reflection.B * weight);

            r = Math.Clamp(r, 0, 255);
            g = Math.Clamp(g, 0, 255);
            b = Math.Clamp(b, 0, 255);

            return Color.FromArgb(r, g, b);
        }


        private Color AddColors(Color color1, Color color2)
        {
            double r = Math.Clamp((color1.R / 255.0) + (color2.R / 255.0), 0, 1);
            double g = Math.Clamp((color1.G / 255.0) + (color2.G / 255.0), 0, 1);
            double b = Math.Clamp((color1.B / 255.0) + (color2.B / 255.0), 0, 1);

            return Color.FromArgb((int)(r * 255), (int)(g * 255), (int)(b * 255));
        }


        private Color ApplyAmbientLighting(Color baseColor)
        {
            double ambientIntensity = 0.1; // Интенсивность фонового освещения

            double r = baseColor.R / 255.0 * ambientIntensity;
            double g = baseColor.G / 255.0 * ambientIntensity;
            double b = baseColor.B / 255.0 * ambientIntensity;

            return Color.FromArgb((int)(r * 255), (int)(g * 255), (int)(b * 255));
        }


        private bool IsPointVisibleFromLight(Point3D hitPoint, Light light)
        {
            Point3D lightDirection = (light.Position - hitPoint).Normalize();
            Ray shadowRay = new Ray(hitPoint + lightDirection * 1e-4, lightDirection);

            foreach (var triangle in triangles)
            {
                if (triangle.Intersect(shadowRay, out double t) && t > 0 && t < (light.Position - hitPoint).Length())
                {
                    return false; // Точка в тени
                }
            }

            foreach (var sphere in spheres)
            {
                if (sphere.Intersect(shadowRay, out double t) && t > 0 && t < (light.Position - hitPoint).Length())
                {
                    return false; 
                }
            }

            return true; // Точка видима
        }

        private Color CalculateLocalLighting(Material material, Point3D normal, Point3D position, Light light)
        {
            normal = normal.Normalize();
            Point3D lightDir = Point3D.Vector(position, light.Position).Normalize();

            // Диффузное освещение
            double normalDotLight = Point3D.DotProduct(normal, lightDir);
            double diff = Math.Max(normalDotLight, 0.0);

            Point3D viewDir = Point3D.Vector(position, camera.Position).Normalize();

            // Расчёт бликов
            double spec = 0.0;
            if(material.Shininess > 0)
            {
                Point3D halfwayDir = (lightDir + viewDir).Normalize();
                spec = Math.Pow(Math.Max(Point3D.DotProduct(normal, halfwayDir), 0.0), material.Shininess);
            }
            
            double materialR = material.DiffuseColor.R / 255.0;
            double materialG = material.DiffuseColor.G / 255.0;
            double materialB = material.DiffuseColor.B / 255.0;

            double lightR = light.Color.R / 255.0;
            double lightG = light.Color.G / 255.0;
            double lightB = light.Color.B / 255.0;

            // Расчёт цвета
            double r = Math.Clamp(materialR * (diff + spec) * lightR * light.Intensity, 0, 1);
            double g = Math.Clamp(materialG * (diff + spec) * lightG * light.Intensity, 0, 1);
            double b = Math.Clamp(materialB * (diff + spec) * lightB * light.Intensity, 0, 1);

            return Color.FromArgb((int)(r * 255), (int)(g * 255), (int)(b * 255));
        }
    }
}


